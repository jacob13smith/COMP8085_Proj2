import json
import torch
from Review import Review
from transformers import DistilBertTokenizerFast, DistilBertModel
import math

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')
    model = DistilBertModel.from_pretrained('distilbert-base-cased')
    reviews = []
    # Will need to grab file name from the user
    with open('training_set.json', 'r') as train:
        for entry in train:
            review = Review(json.loads(entry))
            reviews.append(review)
    print(len(reviews))
    n_reviews_processed = 0
    for i in reviews:
        if n_reviews_processed % 20 == 0:
            print(n_reviews_processed)
        process_review(i, tokenizer, model)
        n_reviews_processed += 1
    return 0

def process_review(review, tokenizer, model):
    tokenized_text = tokenizer.encode(review.text)
    if len(tokenized_text) <= 512:
        with torch.no_grad():
            review.embedded_text = model.forward(torch.tensor([tokenized_text]))[0][0]
    else:
        len_tokenized_text = len(tokenized_text)
        n_extra_tokens = len_tokenized_text - 512
        n_extra_calls = math.ceil(n_extra_tokens/256)
        avg_interval = n_extra_tokens/n_extra_calls
        with torch.no_grad():
            sample_text = torch.tensor([tokenized_text[:512]])
            review.embedded_text = model.forward(sample_text)[0][0][:round(256 + 0.5 * avg_interval)]
            for i in range(1, n_extra_calls):
                i_tokenized_text = round(avg_interval * i)
                j_tokenized_text = round(avg_interval * i + 512)
                sample_text = torch.tensor([tokenized_text[i_tokenized_text:j_tokenized_text]])
                i_sample_text = round(256 + avg_interval * (i - 0.5)) - i_tokenized_text
                j_sample_text = round(256 + avg_interval * (i + 0.5)) - i_tokenized_text
                review.embedded_text = torch.cat((review.embedded_text, model.forward(sample_text)[0][0][i_sample_text:j_sample_text]), 0)
            i_tokenized_text = len_tokenized_text - 512
            sample_text = torch.tensor([tokenized_text[i_tokenized_text:]])
            review.embedded_text = torch.cat((review.embedded_text, model.forward(sample_text)[0][0][round(256 + avg_interval * (n_extra_calls - 0.5)) - i_tokenized_text:]), 0)
    return

if __name__ == '__main__':
    main()