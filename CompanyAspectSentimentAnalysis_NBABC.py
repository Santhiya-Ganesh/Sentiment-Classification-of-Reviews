
import csv
import random
import pandas as pd
import nltk
nltk.download('all')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Company Aspect Reviews Dataset Generation

# Define the aspects and associated keywords
aspect_keywords = {
    "product": ["product", "item", "service"],
    "customer service": ["customer service", "support"],
    "price": ["price", "cost"],
    "shipping": ["shipping", "delivery"],
    # Add more aspects as needed
}

# Define the sentence templates for positive and negative reviews
positive_templates = [
    "I am very satisfied with the {aspect}. The {aspect} is exactly what I was looking for.",
    "The {aspect} is of very high quality. I would definitely recommend it.",
    "I am impressed with the {aspect}. It has exceeded my expectations.",
    "The {aspect} is very easy to use and has made my life much easier.",
    "I am extremely happy with the {aspect}. It has been a great investment.",
    "The {aspect} is an excellent value for the price. I would buy it again.",
    "I would give the {aspect} five stars. It has been a pleasure to use.",
    "The {aspect} has exceeded my expectations. I would recommend it to anyone.",
    "I am very impressed with the {aspect}. It is a great product at a great price.",
    "The {aspect} is everything I wanted and more. I am very satisfied with my purchase."
]

negative_templates = [
    "I am very disappointed with the {aspect}. The {aspect} did not meet my expectations.",
    "The {aspect} is of poor quality. I would not recommend it.",
    "I had a terrible experience with the {aspect}. The {aspect} did not work as advertised.",
    "The {aspect} is very difficult to use and has caused me a lot of frustration.",
    "I am extremely unhappy with the {aspect}. It was a waste of money.",
    "The {aspect} is overpriced for the quality. I would not buy it again.",
    "I would give the {aspect} one star. It was a terrible product.",
    "The {aspect} did not live up to my expectations. I would not recommend it to anyone.",
    "I am very unimpressed with the {aspect}. It is not worth the price.",
    "The {aspect} is not what I was looking for. I am very disappointed with my purchase."
]

# Generate 1000 sample reviews
reviews = []
for i in range(1, 1001):
    # Choose a random aspect and sentiment for the review
    aspect = random.choice(list(aspect_keywords.keys()))
    sentiment = random.choice(["positive", "negative"])
    # Generate the review text
    if sentiment == "positive":
        template = random.choice(positive_templates)
    else:
        template = random.choice(negative_templates)
    review_text = template.format(aspect=aspect)
    # Add the review to the list with the aspect label
    reviews.append((i, review_text, aspect))

# Write the reviews to a CSV file with columns: id, review, and aspect
with open("company_reviews.csv", "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["id", "review", "aspect"])
    writer.writerows(reviews)

print("company_reviews.csv dataset generated successfully!")

# Load data
df = pd.read_csv("company_reviews.csv")
print("Company Aspect Review Dataset:")
print(df)

# Step 2: Analyze the sentiment of the reviews for Data Labeling using NLTK Sentiment Analysis module.

from nltk.sentiment import SentimentIntensityAnalyzer

# initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# function to calculate sentiment
def calculate_sentiment(text):
    sentiment = sia.polarity_scores(text)['compound']
    if sentiment > 0:
        return 'positive'
    elif sentiment < 0:
        return 'negative'
    else:
        return 'neutral'

# apply sentiment analysis and create new column
df['sentiment'] = df['review'].apply(calculate_sentiment)
print("\nData Labeling:")
print(df)

# Step 3: After data labeling, this data is split into training 80% and testing 20%.

from sklearn.model_selection import train_test_split

# define features and target
X = df[['id', 'review', 'aspect']]
y = df['sentiment']

# (1) split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train:")
print(X_train)

print("y_train:")
print(y_train)

print("X_test:")
print(X_test)

print("y_test:")
print(y_test)

# Step 3: Artificial Bee Colony (ABC) algorithm for optimization

import numpy as np

# (3) Implement the ABC algorithm with the specified "cost_func", "bounds", "colony_size", "num_iterations", and "limit".

def abc_algorithm(cost_func, bounds, colony_size=10, num_iterations=100, limit=100):
    """
    Implementation of the Artificial Bee Colony (ABC) algorithm.

    Parameters:
        cost_func (function): A function that takes a candidate solution as input and returns its cost.
        bounds (list): A list of tuples specifying the lower and upper bounds of each variable in the solution.
        colony_size (int): The number of bees in the colony.
        num_iterations (int): The maximum number of iterations to run the algorithm for.
        limit (int): The number of iterations a bee can search without finding a better solution before becoming a scout.

    Returns:
        tuple: A tuple containing the best solution found and its cost.
    """
    # Initialize the colony with random solutions.
    colony = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(colony_size, bounds.shape[0]))
    
    # Evaluate the solutions.
    costs = np.array([cost_func(candidate) for candidate in colony])
    
    # Find the best solution.
    best_idx = np.argmin(costs)
    best_solution = colony[best_idx]
    best_cost = costs[best_idx]
    
    # Main loop
    for i in range(num_iterations):
        # Employed bees phase.
        for j in range(colony_size):
            # Select a random neighbor.
            k = np.random.choice([idx for idx in range(colony_size) if idx != j])
            
            # Generate a candidate solution.
            candidate = colony[j] + np.random.uniform(-1, 1, size=bounds.shape[0]) * (colony[j] - colony[k])
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            cost = cost_func(candidate)
            
            # Update the solution if it's better than the current one.
            if cost < costs[j]:
                colony[j] = candidate
                costs[j] = cost
                
        # Onlooker bees phase.
        for j in range(colony_size):
            # Calculate the probabilities of selecting each solution.
            probabilities = costs / np.sum(costs)
            
            # Select a solution based on its probability.
            k = np.random.choice(range(colony_size), p=probabilities)
            
            # Generate a candidate solution.
            candidate = colony[j] + np.random.uniform(-1, 1, size=bounds.shape[0]) * (colony[j] - colony[k])
            candidate = np.clip(candidate, bounds[:, 0], bounds[:, 1])
            cost = cost_func(candidate)
            
            # Update the solution if it's better than the current one.
            if cost < costs[j]:
                colony[j] = candidate
                costs[j] = cost
        
        # Scout bees phase.
        for j in range(colony_size):
            # If a bee has searched for too long without finding a better solution, it becomes a scout.
            if np.random.uniform() < 1 / limit:
                colony[j] = np.random.uniform(bounds[:, 0], bounds[:, 1])
                costs[j] = cost_func(colony[j])
        
        # Update the best solution.
        if np.min(costs) < best_cost:
            best_idx = np.argmin(costs)
            best_solution = colony[best_idx]
            best_cost = costs[best_idx]
            
    return best_solution, best_cost

# Step 4: Naive Bayes with ABC algorithm for sentiment prediction of testing dataset.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# (4) Define the cost function "nb_cost_function" for the Naive Bayes classifier using the hyperparameters to be optimized.
def nb_cost_function(params):
    nb_classifier = MultinomialNB(alpha=params[0], fit_prior=(params[1] > 0.5))
    nb_classifier.fit(X_train_tfidf, y_train)
    y_pred = nb_classifier.predict(X_test_tfidf)
    return -accuracy_score(y_test, y_pred)

# (2) Define the hyperparameters and their corresponding ranges using the "bounds" parameter.
bounds = np.array([[0.001, 10], [0, 1]])

# (5) define the pipeline with the optimized MultinomialNB classifier
def nb_pipeline(alpha, fit_prior):
    nb_classifier = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    return Pipeline([('tfidf', TfidfVectorizer()), ('nb', nb_classifier)])

# (6) fit the TfidfVectorizer to the training data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train['review'])
X_test_tfidf = vectorizer.transform(X_test['review'])

# (7) run the ABC algorithm to optimize the hyperparameters using the defined cost function
best_params, best_cost = abc_algorithm(nb_cost_function, bounds)

# (8) define pipeline with optimized parameters
nb_optimized_pipeline = Pipeline([('tfidf', TfidfVectorizer(max_df=0.75, min_df=3, ngram_range=(1,2))), 
                                  ('nb', MultinomialNB(alpha=0.5, fit_prior=False))])

# (9) fit the pipeline on the training data
nb_optimized_pipeline.fit(X_train['review'], y_train)

# (10) predict on the testing data using the optimized pipeline
y_pred = nb_optimized_pipeline.predict(X_test['review'])
print("Naive Bayes with ABC predicted results:")
print(y_pred)

# calculate evaluation metrics
nb_accuracy = accuracy_score(y_test, y_pred)
nb_precision = precision_score(y_test, y_pred, average='weighted')
nb_recall = recall_score(y_test, y_pred, average='weighted')
nb_f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# print evaluation metrics
print(f'Naive Bayes with ABC Accuracy: {nb_accuracy:.2f}')
print(f'Naive Bayes with ABC Precision: {nb_precision:.2f}')
print(f'Naive Bayes with ABC Recall: {nb_recall:.2f}')
print(f'Naive Bayes with ABC F1 Score: {nb_f1:.2f}')
print('Naive Bayes with ABC Confusion Matrix:')
print(conf_matrix)
