import json
from Review import Review
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pickle

def test_model(reviews, clf, vectorizer):
    text = [review.text for review in reviews]
    true_stars = [review.stars for review in reviews]
    text_vectors = vectorizer.transform(text)
    predicted_stars = clf.predict(text_vectors)

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

def main():
    # Load test dataset
    reviews = []
    with open('test_set.json', 'r', encoding="utf-8") as test, open('model.pickle', 'rb') as model_file, open('vectorizer.pickle', 'rb') as vectorizer_file:
        total = 0
        for entry in test:
            total += 1
            try:
                review = Review(json.loads(entry))
                reviews.append(review)
            except:
                pass

        # Test the trained model
        clf = pickle.load(model_file)
        vectorizer = pickle.load(vectorizer_file)
        test_model(reviews, clf, vectorizer)
    return 0

if __name__ == '__main__':
    main()