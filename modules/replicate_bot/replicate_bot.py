from modules.input_data_creator.Delusion import Delusion
from modules.input_data_creator.Obama import Obama
from modules.input_data_creator.Potato import Potato
from modules.input_data_creator.USFCA import USFCA
from textblob import TextBlob  # For sentiment analysis
from collections import Counter  # For word frequency
import csv
import os
import random
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from sentence_transformers import SentenceTransformer  # For using pre-trained embeddings
from sklearn.metrics.pairwise import cosine_similarity

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

    def __init__(self, person, train_model=True):
        self.tokenized_dataset = None
        self.training_data = None
        self.data = None
        self.person = person
        self.model_name = f"model_{self.person}.pt"

        print("Initializing model param")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

        if not train_model:
            return

        print("Parsing input data")
        self.create_data()
        self.create_features()
        self.save_training_data_to_csv()

        self.validate_train_data()

    def create_data(self):
        if self.person == "Obama":
            self.data = Obama().create()

        if self.person == "Potato":
            self.data = Potato().create()

        if self.person == "USFCA":
            self.data = USFCA().create()

        if self.person == "Delusion":
            self.data = Delusion().create()

    def save_model(self):
        folder_path = "output/models"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        model_path = os.path.join(folder_path, self.model_name)
        torch.save(self.model.state_dict(), model_path)

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

    def validate_train_data(self):
        if not self.training_data:
            exit()

    def get_embedding(self, text):
        return self.embedder.encode(text)

    def compute_semantic_similarity(self, embedding1, embedding2):
        # Compute cosine similarity between embeddings
        similarity_matrix = cosine_similarity([embedding1], [embedding2])
        return similarity_matrix[0][0]

    def compute_loss(self, generated_responses, ground_truth_responses):
        similarity_score = 0
        for gen_resp, gt_resp in zip(generated_responses, ground_truth_responses):
            # Compute semantic similarity based on embeddings
            gen_embedding = self.get_embedding(gen_resp)
            gt_embedding = self.get_embedding(gt_resp)
            similarity_score += self.compute_semantic_similarity(gen_embedding, gt_embedding)

        # Convert similarity_score to a PyTorch tensor
        loss = torch.tensor(similarity_score, dtype=torch.float32, requires_grad=True)
        return loss

    def train(self, num_epochs, train_data, valid_data):
        iter_count = 0
        print("Number of num_epochs : "+str(num_epochs) )
        for epoch in range(num_epochs):
            # Training loop
            print("Epoch : "+str(epoch))
            print("Training model")
            self.model.train()  # Set the model to training mode
            total_train_loss = 0
            random.shuffle(train_data)  # Shuffle the training data
            batch_count = 0
            iter_count = iter_count + 1
            for batch in train_data:
                print("Batch count start : " + str(batch_count)+ " Iter Count : "+str(iter_count))
                generated_responses = self.generate_responses(batch["user"])
                loss = self.compute_loss(generated_responses, batch["bot"])

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_train_loss += loss.item()
                print("Batch count end : "+str(batch_count)+ " Iter Count : "+str(iter_count))
                batch_count=batch_count+1

            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            total_valid_loss = 0
            with torch.no_grad():
                for batch in valid_data:
                    generated_responses = self.generate_responses(batch["user"])
                    loss = self.compute_loss(generated_responses, batch["bot"])
                    total_valid_loss += loss.item()

            # Compute average losses
            avg_train_loss = total_train_loss / len(train_data)
            avg_valid_loss = total_valid_loss / len(valid_data)

            # Print training and validation losses
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")

    def generate_responses(self, user_inputs):
        # Generate responses from user inputs using the model
        generated_responses = []
        for user_input in user_inputs:
            # Tokenize user input
            input_ids = self.tokenizer.encode(user_input, return_tensors="pt", max_length=15, truncation=True)

            # Generate response
            output_ids = self.model.generate(input_ids, max_length=15, num_return_sequences=1,
                                             pad_token_id=self.tokenizer.eos_token_id)

            # Decode generated response
            generated_response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            generated_responses.append(generated_response)

        return generated_responses

    def emulate(self):
        while True:
            user_input = input("User: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            bot_responses = self.generate_responses(user_input)
            for bot_response in bot_responses:
                print("Bot:", bot_response)


def main():
    # Initialize ReplicateBot with person
    person = "USFCA"  # Change this to the desired person
    bot = ReplicateBot(person)

    # Create data for training and validation
    bot.create_data()

    # Split data into training and validation sets
    data = bot.data
    num_samples = len(data)
    train_ratio = 0.8  # 80% training, 20% validation
    num_train_samples = int(train_ratio * num_samples)

    train_data = data[:num_train_samples]
    valid_data = data[num_train_samples:]

    # Train the model
    num_epochs = 1  # Adjust the number of epochs as needed
    bot.train(num_epochs, train_data, valid_data)
    bot.save_model()
    bot.emulate()

if __name__ == "__main__":
    main()
