import json
from Review import Review

def main():
    reviews = []
    # Will need to grab file name from the user
    with open('training_set.json', 'r') as train:
        for entry in train:
            review = Review(json.loads(entry))
            reviews.append(review)
    print(len(reviews))
    return 0

if __name__ == '__main__':
    main()