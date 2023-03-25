import json

def main():
    text, stars, useful, funny, cool = [], [], [], [], []
    n_reviews_processed = 0
    with open('yelp_academic_dataset_review.json') as f:
        for line in f:
            print(n_reviews_processed)
            if n_reviews_processed <= 189500:
                review = json.loads(line)
                if n_reviews_processed == 5:
                    review = {}
                processed_review = process_review(review)
                text.append(processed_review["text"])
                stars.append(processed_review["stars"])
                useful.append(processed_review["useful"])
                funny.append(processed_review["funny"])
                cool.append(processed_review["cool"])
            else:
                break
            n_reviews_processed += 1
    print(text[5])
    print(stars[5])
    print(useful[5])
    print(funny[5])
    print(cool[5])
    return 0

def process_review(review):
    rv = {}
    if "text" in review:
        rv["text"] = review["text"]
    else:
        rv["text"] = None
    if "stars" in review:
        rv["stars"] = review["stars"]
    else:
        rv["stars"] = None
    if "useful" in review:
        rv["useful"] = review["useful"]
    else:
        rv["useful"] = None
    if "funny" in review:
        rv["funny"] = review["funny"]
    else:
        rv["funny"] = None
    if "cool" in review:
        rv["cool"] = review["cool"]
    else:
        rv["cool"] = None
    return rv

if __name__ == '__main__':
    main()