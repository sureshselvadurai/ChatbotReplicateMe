from modules.replicate_bot import ReplicateBot

if __name__ == '__main__':

    # persons = ["Delusion","USFCA","Potato","Obama"]
    persons = ["USFCA"]
    for person in persons:
        bot = ReplicateBot(person)
        bot.create_data()
        # bot.create_features()

        # Split data into training and validation sets
        data = bot.data
        num_samples = len(data)
        train_ratio = 0.8  # 80% training, 20% validation
        num_train_samples = int(train_ratio * num_samples)

        train_data = data[:num_train_samples]
        valid_data = data[num_train_samples:]

        # Train the model
        num_epochs = 25  # Adjust the number of epochs as needed
        bot.train(num_epochs, train_data, valid_data)
        bot.save_model()

