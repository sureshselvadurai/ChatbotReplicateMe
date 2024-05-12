from modules.utils.utils import merge_txt_files
import os


def parse_text_file(file_path, min_word_count=4):
    questions_answers = []
    current_question = None

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("Q"):
                # Found a new question
                if current_question and len(current_question["user"].split()) >= min_word_count:
                    # Append the previous question and answer to the list if its word count is >= min_word_count
                    questions_answers.append(current_question)
                # Start a new question
                current_question = {"user": line[3:].strip(), "bot": ""}
            elif line.startswith("MR. EARNEST:") or line.startswith("MR:RAJA:") or current_question:
                # Found an answer line or continuation of the question or answer
                if current_question:
                    if line.startswith("MR. EARNEST:") or line.startswith("MR:RAJA:"):
                        line = line.split(":", 1)[-1].strip()
                    current_question["bot"] += line + "\n"  # Append to existing question or answer
                else:
                    # If there's no current question, it's likely a continuation of the previous answer
                    print("Warning: Found an answer without a corresponding question.")
            elif line:  # Check if the line is not empty
                # If there's no current question and the line is not empty, it's likely a multi-line question
                current_question = {"user": line.strip(), "bot": ""}

    # Append the last question and answer to the list if its word count is >= min_word_count
    if current_question and len(current_question["user"].split()) >= min_word_count:
        questions_answers.append(current_question)

    return questions_answers



def parse_folder(folder):
    questions_answers = []

    # Iterate through each file in the folder
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        # Check if the file is a regular file (not a directory)
        if os.path.isfile(file_path):
            # Parse the text file and append its questions and answers to the list
            questions_answers.extend(parse_text_file(file_path))

    return questions_answers


class Obama:

    def __init__(self):
        self.data = None

    def create(self):
        input_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../input/Obama"))
        self.data = parse_folder(input_directory)
        return self.data

def main():
    obama_instance = Obama()
    obama_data = obama_instance.create()
    # Do something with obama_data


if __name__ == "__main__":
    main()