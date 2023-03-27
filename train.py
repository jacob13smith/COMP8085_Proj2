import json
from Review import Review
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pickle
import argparse

techniques = ['bert', 'dtc', 'svm']

def main():
    parser = argparse.ArgumentParser(
                    prog='Project 2 - Model Training Program',
                    description='Program to train models for predicting Yelp review stars')
    parser.add_argument('technique', choices=techniques,
                    help='Which technique to train the model for.')
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
            if total >= 1000000: break
            total += 1
            try:
                review = Review(json.loads(entry))
                reviews.append(review)
                processed += 1
            except:
                skipped += 1

    if args.technique == 'bert':
        # BERT code
        pass
    elif args.technique == 'dtc':
        # Decision Tree Classifier code

        print('Vectorizing words found in all review texts...\n')
        # Train the model(s) below with the reviews[] array
        vectorizer = TfidfVectorizer(max_features=20000)
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
        with open('dtc_model.pickle', 'wb') as f:
            pickle.dump(model, f)
        with open('dtc_vectorizer.pickle', 'wb') as f:
            pickle.dump(vectorizer, f)

        print('Decision Tree Classifier model trained and ready to use!')
        pass
    elif args.technique == 'svm':
        # SVM code
        # Split the data into text and labels
        text = [review.text for review in reviews]
        stars = [review.stars for review in reviews]
        useful = [review.useful for review in reviews]
        funny = [review.funny for review in reviews]
        cool = [review.cool for review in reviews]

        # Convert the text into a bag-of-words representation
        stopWords = ['english', 'french', 'spanish', 'russian']
        vectorizer = CountVectorizer(stop_words= stopWords)
        X = vectorizer.fit_transform(text)

        # Train SVM models to predict star ratings, usefulness, funniness, and coolness
        clf_stars = SVC(kernel='linear', random_state=42)
        clf_stars.fit(X, stars)

        clf_useful = SVC(kernel='linear', random_state=42)
        clf_useful.fit(X, useful)

        clf_funny = SVC(kernel='linear', random_state=42)
        clf_funny.fit(X, funny)

        clf_cool = SVC(kernel='linear', random_state=42)
        clf_cool.fit(X, cool)

        # Save the trained models and vectorizer to output files
        with open('svm.pickle', 'wb') as f:
            pickle.dump((clf_stars, clf_useful, clf_funny, clf_cool, vectorizer), f)
        pass


    return 0

if __name__ == '__main__':
    main()