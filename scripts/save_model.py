import os
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Define the workspace path dynamically
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
save_path = os.path.join(workspace_dir, 'trained_model')

# Load the tokenizer and initialize model architecture
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = TFT5ForConditionalGeneration.from_pretrained('t5-small')

# Save the model using Hugging Face's `save_pretrained()` for compatibility
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"Model and tokenizer saved to {save_path}")
