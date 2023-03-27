import argparse

def main():
    parser = argparse.ArgumentParser(
                    prog='Yelp Data splitter',
                    description='Split Yelp academic review dataset into training and test')
    parser.add_argument('training_size', type=float,
                    help='A decimal number representing percentage of original data to be used for training (between 0.0 and 1.0)')
    args = parser.parse_args()

    total_entries = sum(1 for _ in open('yelp_academic_dataset_review.json', encoding="utf-8"))
    current_entry_index = 0
    with open('yelp_academic_dataset_review.json', encoding="utf-8") as input, open('training_set.json', 'w', encoding="utf-8") as train, open('test_set.json', 'w', encoding="utf-8") as test:
        for line in input:
            if current_entry_index < total_entries * args.training_size:
                train.write(line)
            else:
                test.write(line)
            current_entry_index += 1
    print("Total records: " + str(current_entry_index))
    print("Training records: " + str(int(total_entries * args.training_size)))
    print("Testing records: " + str(int(current_entry_index - total_entries * (args.training_size))))
    return


if __name__ == '__main__':
    main()