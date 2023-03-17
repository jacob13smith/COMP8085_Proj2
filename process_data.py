import json

def main():
    data = []
    with open('yelp_academic_database.json') as f:
        for line in f:
            data.append(json.loads(line))   
    return 0

if __name__ == 'main':
    main()