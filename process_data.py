import json

def main():
    data = []
    with open('yelp_academic_dataset_review.json') as f:
        for line in f:
            data.append(json.loads(line))
    print(len(data))
    return 0

if __name__ == '__main__':
    main()