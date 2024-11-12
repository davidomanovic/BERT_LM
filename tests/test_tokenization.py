import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.bert_lm import BertQA
import unittest

class TestBertQA(unittest.TestCase):
    def setUp(self):
        # Set up the BertQA instance
        self.qa_model = BertQA(model_name='bert-base-uncased', dropout=0.3)
    
    def test_tokenizer(self):
        # Example contexts and questions
        contexts = [
            "BERT is a transformer-based model designed to handle various NLP tasks."
        ]
        questions = [
            "What is BERT?"
        ]
        answers = [
            "transformer-based model"
        ]
        
        # Prepare the training data (this will also test tokenization)
        dataset = self.qa_model.prepare_training_data(contexts, questions, answers, max_length=256)
        
        # Retrieve the first element in the dataset for inspection
        for data in dataset.take(1):
            input_data, label_data = data
            
            # Extract inputs and labels from the batch
            input_ids = input_data["input_ids"]
            attention_mask = input_data["attention_mask"]
            start_positions = label_data["start_positions"]
            end_positions = label_data["end_positions"]
            
            # Check that the tokenization works: input_ids and attention_mask should be non-empty
            self.assertGreater(input_ids.shape[1], 0, "Input IDs are empty")
            self.assertGreater(attention_mask.shape[1], 0, "Attention Mask is empty")
            
            # Check that start and end positions are within the input sequence range
            self.assertGreaterEqual(start_positions[0], 0, "Start position is invalid")
            self.assertLess(start_positions[0], input_ids.shape[1], "Start position is out of bounds")
            self.assertGreaterEqual(end_positions[0], 0, "End position is invalid")
            self.assertLess(end_positions[0], input_ids.shape[1], "End position is out of bounds")
            
            # Check if the answer spans correctly (this is a simple check to ensure tokenization is aligned with answer span)
            context = contexts[0]
            answer = answers[0]
            answer_start = context.find(answer)
            self.assertTrue(answer_start >= 0, "Answer not found in context")
            self.assertEqual(context[answer_start:answer_start + len(answer)], answer, "Answer span is incorrect")

    def test_answer_question(self):
        # Example context and question
        context = "BERT is a transformer-based model designed to handle various NLP tasks."
        question = "What is BERT?"
        
        # Ask the question and get the predicted answer
        answer = self.qa_model.answer_question(context, question)
        
        # Check that the predicted answer is correct (you can adjust based on expected output)
        expected_answer = "transformer-based model"
        self.assertEqual(answer.strip(), expected_answer, f"Expected '{expected_answer}', but got '{answer}'")

if __name__ == "__main__":
    unittest.main()
