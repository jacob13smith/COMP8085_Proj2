import json
from Review import Review

def main():
    reviews = []
    # Will need to grab file name from the user
    with open('training_set.json', 'r', encoding="utf-8") as train:
        total = 0
        processed = 0
        skipped = 0
        for entry in train:
            total += 1
            if total > 10: break
            try:
                review = Review(json.loads(entry))
                reviews.append(review)
                processed += 1
            except:
                skipped += 1

    print("Total reviews processed: " + str(total))
    print("Total reviews skipped: " + str(skipped))
    print("Total useful reviews: " + str(processed))



    return 0

if __name__ == '__main__':
    main()