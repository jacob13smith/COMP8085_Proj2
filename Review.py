class Review:
    def __init__(self, data_dict):
        if data_dict.get('stars') is None:
            print('No stars!')
        
        if data_dict.get('text') is None:
            print('No text!')
        
        if data_dict.get('funny') is None:
            print('No funny!')

        if data_dict.get('cool') is None:
            print('No cool!')
        
        if data_dict.get('useful') is None:
            print('No useful!')

        self.review_id = data_dict.get('review_id')
        self.stars = data_dict.get('stars')
        self.useful = data_dict.get('useful')
        self.funny = data_dict.get('funny')
        self.cool = data_dict.get('cool')
        self.text = data_dict.get('text')
        self.date = data_dict.get('date')

    def __str__(self):
        output = "{\nReview ID: " + self.review_id + "\nStars: " + str(self.stars) + "\nUseful: " + str(self.useful) + "\nFunny: " + str(self.funny) + "\nCool: " + str(self.cool) + "\nText: " + self.text + "\nDate: "+ self.date + "\n}"
        return output
