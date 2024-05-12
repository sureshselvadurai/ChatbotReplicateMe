import csv
import os


def parse_csv_file(file_path):
    questions_answers = []

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            department = row['Department']
            question = row['Question']
            answer = row['Answer']
            questions_answers.append({"user": f"{department}: {question}", "bot": answer})

    return questions_answers


class Delusion:

    def __init__(self):
        self.data = None

    def create(self):
        input_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../input/Delusion/DelusionData.csv"))
        self.data = parse_csv_file(input_directory)
        return self.data
