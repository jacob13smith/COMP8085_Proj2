class Review:
    def __init__(self, data_dict, enforce_fields=True):
        if enforce_fields and None in [data_dict.get('text'), data_dict.get('funny'), data_dict.get('cool'), data_dict.get('useful')] or data_dict.get('text') == "":
            print("skipping")
            raise Exception("Missing necessary field(s)")

        self.review_id = data_dict.get('review_id')
        self.stars = data_dict.get('stars')
        self.useful = data_dict.get('useful')
        self.funny = data_dict.get('funny')
        self.cool = data_dict.get('cool')
        self.text = data_dict.get('text')

    def __str__(self):
        output = "{\nReview ID: " + self.review_id + "\nStars: " + str(self.stars) + "\nUseful: " + str(self.useful) + "\nFunny: " + str(self.funny) + "\nCool: " + str(self.cool) + "\nText: " + self.text + "\n}"
        return output
