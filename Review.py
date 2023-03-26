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
<<<<<<< HEAD
=======
        self.date = data_dict.get('date')
        self.embedded_text = None

>>>>>>> 9c38f15 (contextual embeddings)

    def __str__(self):
        output = "{\n    Review ID: " + self.review_id + "\n    Stars: " + str(self.stars) + "\n    Useful: " + str(self.useful) + "\n    Funny: " + str(self.funny) + "\n    Cool: " + str(self.cool) + "\n    Text: " + self.text + "\n}"
        return output
