import unittest
import tensorflow as tf
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.bert_lm import BertQA  # Import the BertQA class

class TestBertQA(unittest.TestCase):

    def setUp(self):
        # Initialize the BertQA class
        self.qa_model = BertQA(model_name='bert-base-uncased', dropout=0.3)
        
    def test_model_initialization(self):
        # Test that the model and tokenizer initialize without errors
        self.assertIsNotNone(self.qa_model.model, "Model not initialized correctly")
        self.assertIsNotNone(self.qa_model.tokenizer, "Tokenizer not initialized correctly")

    def test_model_output_shape(self):
        # Example context and question
        context = "BERT is a transformer-based model designed to handle various NLP tasks."
        question = "What is BERT?"
        
        # Tokenize and run the model to check the output shape
        inputs = self.qa_model.tokenizer(question, context, return_tensors="tf", truncation=True)
        outputs = self.qa_model.model(inputs)

        # Check that the model outputs start_logits and end_logits
        self.assertIn("start_logits", outputs, "start_logits not in output")
        self.assertIn("end_logits", outputs, "end_logits not in output")
        
        # Ensure that start_logits and end_logits are of correct shape
        self.assertEqual(outputs.start_logits.shape[0], 1, "Batch size is incorrect")
        self.assertEqual(outputs.end_logits.shape[0], 1, "Batch size is incorrect")
        self.assertEqual(outputs.start_logits.shape[1], inputs["input_ids"].shape[1], "Start logits shape is incorrect")
        self.assertEqual(outputs.end_logits.shape[1], inputs["input_ids"].shape[1], "End logits shape is incorrect")

    def test_edge_case_empty_context(self):
        # Edge case with empty context
        context = ""
        question = "What is BERT?"
        
        # Ask the question with an empty context
        answer = self.qa_model.answer_question(context, question)
        
        # Check if the answer is empty or a reasonable fallback (can be adjusted based on model behavior)
        self.assertEqual(answer.strip(), "", "Answer should be empty for empty context")

    def test_edge_case_empty_question(self):
        # Edge case with empty question
        context = "BERT is a transformer-based model designed to handle various NLP tasks."
        question = ""
        
        # Ask an empty question
        answer = self.qa_model.answer_question(context, question)
        
        # Check if the answer is empty or a reasonable fallback
        self.assertEqual(answer.strip(), "", "Answer should be empty for empty question")

    def test_model_save_and_load(self):
        # Test saving and loading the model and tokenizer
        save_dir = "./saved_model"
        
        # Save the model
        self.qa_model.save_model(save_dir)
        
        # Check if the model directory is created
        self.assertTrue(os.path.exists(save_dir), "Model saving failed")
        
        # Load the saved model and tokenizer
        new_model = BertQA(model_name=save_dir)
        
        # Check that the new model and tokenizer are loaded correctly
        self.assertIsNotNone(new_model.model, "Loaded model is None")
        self.assertIsNotNone(new_model.tokenizer, "Loaded tokenizer is None")

        # Clean up saved model
        os.rmdir(save_dir)

if __name__ == "__main__":
    unittest.main()
