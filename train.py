import json
from Review import Review
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
import pickle

def main():
    reviews = []
    # Will need to grab file name from the user
    with open('training_set.json', 'r', encoding="utf-8") as train:
        total = 0
        processed = 0
        skipped = 0
        for entry in train:
            if total >= 100000: break
            total += 1
            try:
                review = Review(json.loads(entry))
                reviews.append(review)
                processed += 1
            except:
                skipped += 1

    print("Total reviews processed: " + str(total))
    print("Total reviews skipped: " + str(skipped))
    print("Total useful reviews: " + str(processed))

    # Train the model(s) below with the reviews[] array
    vectorizer = TfidfVectorizer(max_features=10000)
    text_vectors = vectorizer.fit_transform([review.text for review in reviews])
    stars = [review.stars for review in reviews]

    model = DecisionTreeClassifier()
    model.fit(text_vectors, stars)

    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pickle', 'wb') as f:
        pickle.dump(vectorizer, f)

    return 0

if __name__ == '__main__':
    main()