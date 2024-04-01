from datasets import Dataset

from modules.input_data_creator.Delusion import Delusion
from modules.input_data_creator.Obama import Obama
from modules.input_data_creator.Potato import Potato
from modules.input_data_creator.USFCA import USFCA
from textblob import TextBlob  # For sentiment analysis
from collections import Counter  # For word frequency
import csv
import os


def extract_features(conv):
    # Length of user and bot utterances
    conv['user_length'] = len(conv['user'].split())
    conv['bot_length'] = len(conv['bot'].split())

    # Sentiment analysis of user and bot utterances
    conv['user_sentiment'] = TextBlob(conv['user']).sentiment.polarity
    conv['bot_sentiment'] = TextBlob(conv['bot']).sentiment.polarity

    # Word frequency in user and bot utterances
    user_words = conv['user'].lower().split()
    bot_words = conv['bot'].lower().split()
    conv['user_word_frequency'] = dict(Counter(user_words))
    conv['bot_word_frequency'] = dict(Counter(bot_words))

    return conv

class ReplicateBot:

    def __init__(self, person):
        self.tokenized_dataset = None
        self.training_data = None
        self.train = None
        self.data = None
        self.person = person
        self.create_data()
        self.create_features()
        self.save_training_data_to_csv()

    def create_data(self):
        if self.person == "Obama":
            self.data = Obama().create()

        if self.person == "Potato":
            self.data = Potato().create()

        if self.person == "USFCA":
            self.data = USFCA().create()

        if self.person == "Delusion":
            self.data = Delusion().create()

    def create_features(self):
        conversations_with_features = [extract_features(conv) for conv in self.data]

        # Convert conversation data to a format suitable for training
        self.training_data = {
            "user": [conv['user'] for conv in conversations_with_features],
            "bot": [conv['bot'] for conv in conversations_with_features],
            "user_length": [conv['user_length'] for conv in conversations_with_features],
            "bot_length": [conv['bot_length'] for conv in conversations_with_features],
            "user_sentiment": [conv['user_sentiment'] for conv in conversations_with_features],
            "bot_sentiment": [conv['bot_sentiment'] for conv in conversations_with_features],
            "user_word_frequency": [conv['user_word_frequency'] for conv in conversations_with_features],
            "bot_word_frequency": [conv['bot_word_frequency'] for conv in conversations_with_features]
        }

    def save_training_data_to_csv(self):
        folder_path = "output/features"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filename = os.path.join(folder_path, f"training_data_{self.person}.csv")
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ["user", "bot", "user_length", "bot_length", "user_sentiment", "bot_sentiment",
                          "user_word_frequency", "bot_word_frequency"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i in range(len(self.training_data["user"])):
                row = {field: self.training_data[field][i] for field in fieldnames}
                writer.writerow(row)
