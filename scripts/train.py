import os
import sys
from datasets import load_dataset
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.bert_lm import BertQA
import json

workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
save_path = os.path.join(workspace_dir, 'trained_model')
dataset_path = os.path.join(workspace_dir, 'data', 'dataset.json')

def load_dataset(file_path):
    """Load the dataset from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Extract contexts, questions, and answers for training
    contexts = [item['context'] for item in data]
    questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]
    return contexts, questions, answers

def main():
    # Initialize the BERT QA model
    bert_qa = BertQA()

    # Load the question-answer dataset
    contexts, questions, answers = load_dataset(dataset_path)  # Specify your dataset path

    # Train the model on the dataset
    bert_qa.train(contexts, questions, answers, epochs=10, batch_size=8, max_length=256)

    # Save the trained model
    bert_qa.save_model(save_path)  # Specify your save path

if __name__ == "__main__":
    main()
