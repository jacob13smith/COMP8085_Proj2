import argparse

def main():
    parser = argparse.ArgumentParser(
                    prog='Yelp Data splitter',
                    description='Split Yelp academic review dataset into training and test')
    parser.add_argument('training_size', type=float,
                    help='A decimal number representing percentage of original data to be split into training')
    args = parser.parse_args()

    total_entries = sum(1 for line in open('yelp_academic_dataset_review.json'))
    current_entry_index = 0
    with open('yelp_academic_dataset_review.json') as input, open('training_set.json', 'w') as train, open('test_set.json', 'w') as test:
        for line in input:
            if current_entry_index < total_entries * args.training_size:
                train.write(line)
            else:
                test.write(line)
            current_entry_index += 1
    return


if __name__ == '__main__':
    main()