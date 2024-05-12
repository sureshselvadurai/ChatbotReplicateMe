import torch
from modules.replicate_bot import ReplicateBot


def main():
    # Initialize ReplicateBot with the desired person
    person = "USFCA"  # Change this to the desired person
    bot = ReplicateBot(person,train_model=False)

    # Load the trained model
    model_path = f"output/models/model_{person}20Iter.pt"
    bot.model.load_state_dict(torch.load(model_path))
    bot.model.eval()

    # Run the emulator
    bot.emulate()

if __name__ == "__main__":
    main()
