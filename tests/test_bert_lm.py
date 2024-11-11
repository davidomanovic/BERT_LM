import tensorflow as tf
from transformers import TFBertForQuestionAnswering, BertTokenizerFast
import json
import os
import sys

# Add parent directory to the path (if required for your project structure)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class QAInference:
    def __init__(self, model_path):
        """
        Initializes the model and tokenizer for inference.
        """
        # Load the fine-tuned model and tokenizer
        self.model = TFBertForQuestionAnswering.from_pretrained(model_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)

    def answer_question(self, context, question):
        """
        Given a context and a question, return the answer predicted by the model.
        """
        # Tokenize the input question and context
        inputs = self.tokenizer(question, context, return_tensors="tf", truncation=True, padding=True)
        
        # Get model outputs
        outputs = self.model(inputs)
        
        # Find the start and end logits of the predicted answer
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        # Get the start and end positions of the answer
        start_idx = tf.argmax(start_scores, axis=1).numpy()[0]
        end_idx = tf.argmax(end_scores, axis=1).numpy()[0]
        
        # Extract the answer tokens and decode to string
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return answer

def main():
    # Define the path to the model and dataset
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(workspace_dir, 'trained_model')  # path to model folder (not .h5 directly)
    dataset_path = os.path.join(workspace_dir, 'data', 'dataset.json')  # path to dataset.json file

    # Load dataset from JSON
    with open(dataset_path, 'r') as file:
        dataset = json.load(file)

    # Initialize the QA model
    qa_model = QAInference(model_path)

    # Test the model on each example in the dataset
    for i, data in enumerate(dataset):
        context = data['context']
        question = data['question']
        true_answer = data['answer']
        
        print(f"Example {i + 1}:")
        print(f"Context: {context}")
        print(f"Question: {question}")
        print(f"True Answer: {true_answer}")
        
        # Get the predicted answer from the model
        predicted_answer = qa_model.answer_question(context, question)
        print(f"Predicted Answer: {predicted_answer}")
        print("=" * 50)

if __name__ == "__main__":
    main()
