import json
from Review import Review
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pickle
import argparse

def main():
    parser = argparse.ArgumentParser(
                    prog='Project 2 - Model Training Program',
                    description='Program to train models for predicting Yelp review stars')
    parser.add_argument('-t', dest='enable_tuning', default=False, action='store_true',
                    help='Optional flag to enable hyperparameter tuning during training')
    args = parser.parse_args()

    reviews = []
    # Will need to grab file name from the user
    with open('training_set.json', 'r', encoding="utf-8") as train:
        print('Pre-processing data from training dataset...\n')
        total = 0
        processed = 0
        skipped = 0
        for entry in train:
            if total >= 1000: break
            total += 1
            try:
                review = Review(json.loads(entry))
                reviews.append(review)
                processed += 1
            except:
                skipped += 1
    print('Vectorizing words found in all review texts...\n')
    # Train the model(s) below with the reviews[] array
    vectorizer = TfidfVectorizer(max_features=10000)
    text_vectors = vectorizer.fit_transform([review.text for review in reviews])
    stars = [review.stars for review in reviews]

    if args.enable_tuning:
        print('Tuning hyper parameters...\n')
        param_grid = {'criterion': ['entropy'],
                      'max_depth': [None, 10, 20],
                      'max_leaf_nodes': [None, 20, 40],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]}
        grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
        grid_search.fit(text_vectors, stars)
        model = DecisionTreeClassifier(**grid_search.best_params_)
    else:
        model = DecisionTreeClassifier()

    print('Fitting model to training data...\n')    
    model.fit(text_vectors, stars)

    print('Saving model...')
    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)

    print('Model trained and ready to use!')
    return 0

if __name__ == '__main__':
    main()