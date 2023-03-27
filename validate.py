import json
from Review import Review
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle
import argparse
import math

techniques = ['bert', 'dtc', 'svm']

def main():
    
    parser = argparse.ArgumentParser(
                    prog='Project 2 - Model Training Program',
                    description='Program to train models for predicting Yelp review stars')
    parser.add_argument('technique', choices=techniques,
                    help='Which model to use for validation.')
    parser.add_argument('-t', dest='enable_tuning', default=False, action='store_true',
                    help='Optional flag to enable hyperparameter tuning during training')
    parser.add_argument('--max-validate', dest='max_validate_entries', type=float, default=math.inf,
                    help='Maximum number of reviews to validate the model with.')
    args = parser.parse_args()

    # Load test dataset
    reviews = []
    with open('test_set.json', 'r', encoding="utf-8") as test:
        total = 0
        print('Pre-processing data from validation dataset...\n')
        for entry in test:
            if total >= args.max_validate_entries: break
            total += 1
            try:
                review = Review(json.loads(entry))
                reviews.append(review)
            except:
                pass

    if args.technique == 'bert':
        # BERT validation code
        pass

    elif args.technique == 'dtc':
        # Decision Tree Classifier validation code
        print('Loading trained model from disk...\n')
        with open('dtc_model.pickle', 'rb') as model_file, open('dtc_vectorizer.pickle', 'rb') as vectorizer_file:

            # Test the trained model
            clf = pickle.load(model_file)
            vectorizer = pickle.load(vectorizer_file)
            text = [review.text for review in reviews]
            true_stars = [review.stars for review in reviews]

            print('Vectorizing validation data text fields...\n')
            text_vectors = vectorizer.transform(text)

            print('Predicting stars from vectorized text...\n')
            predicted_stars = clf.predict(text_vectors)

            print('Predictions Complete!\nAnalyzing accuracy...\n')
            # Compute accuracy
            accuracy = accuracy_score(true_stars, predicted_stars)
            print("Accuracy: {:.2f}%".format(accuracy * 100))

            # Print classification report
            report = classification_report(true_stars, predicted_stars)
            print(report)

            # Print confusion matrix
            matrix = confusion_matrix(true_stars, predicted_stars)
            print("Confusion matrix:")
            print(matrix)

    elif args.technique == 'svm':
        # SVM code
        # Split the data into text and labels
        text = [review.text for review in reviews]
        stars = [review.stars for review in reviews]
        useful = [review.useful for review in reviews]
        funny = [review.funny for review in reviews]
        cool = [review.cool for review in reviews]
        # Load the trained models and vectorizer from the input files
        with open('svm.pickle', 'rb') as f:
            clf_stars, clf_useful, clf_funny, clf_cool, vectorizer = pickle.load(f)

        # Convert the test data into a bag-of-words representation using the vectorizer
        X_test = vectorizer.transform(text)

        # Predict the ratings and review attributes using the trained models
        y_pred_stars = clf_stars.predict(X_test)
        y_pred_useful = clf_useful.predict(X_test)
        y_pred_funny = clf_funny.predict(X_test)
        y_pred_cool = clf_cool.predict(X_test)
        
        # Print the classification reports to evaluate the models' performance
        print('Star ratings:')
        print(classification_report(stars, y_pred_stars))

        print('Usefulness:')
        print(classification_report(useful, y_pred_useful))

        print('Funniness:')
        print(classification_report(funny, y_pred_funny))

        print('Coolness:')
        print(classification_report(cool, y_pred_cool))
        pass
    return 0

if __name__ == '__main__':
    main()