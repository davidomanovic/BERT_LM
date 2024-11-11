from transformers import TFBertForQuestionAnswering, BertTokenizerFast
import tensorflow as tf
import numpy as np

class BertQA:
    def __init__(self, model_name='bert-base-uncased', dropout=0.3):
        # Initialize the BERT model and tokenizer for question-answering
        self.model = TFBertForQuestionAnswering.from_pretrained(model_name, hidden_dropout_prob=dropout)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)  # Use BertTokenizerFast

    def prepare_training_data(self, contexts, questions, answers, max_length=256):
        """
        Prepare the dataset for training by tokenizing inputs and matching start/end positions of answers.
        """
        input_ids, attention_masks, start_positions, end_positions = [], [], [], []
        
        for context, question, answer in zip(contexts, questions, answers):
            # Tokenize the context and question
            encoded = self.tokenizer(
                question,
                context,
                truncation="only_second",
                max_length=max_length,
                padding="max_length",
                return_offsets_mapping=True,  # Only available with fast tokenizer
                return_tensors="np"
            )
            
            # Find the start and end token positions for the answer
            start_idx = context.find(answer)
            end_idx = start_idx + len(answer)
            
            offsets = encoded["offset_mapping"][0]
            start_token = end_token = 0

            # Locate start and end positions within the tokenized offsets
            for i, (start, end) in enumerate(offsets):
                if start <= start_idx < end:
                    start_token = i
                if start < end_idx <= end:
                    end_token = i
                    break

            input_ids.append(encoded["input_ids"][0])
            attention_masks.append(encoded["attention_mask"][0])
            start_positions.append(start_token)
            end_positions.append(end_token)

        # Convert lists to numpy arrays and format them as a tf.data.Dataset
        input_ids = np.array(input_ids)
        attention_masks = np.array(attention_masks)
        start_positions = np.array(start_positions)
        end_positions = np.array(end_positions)

        return tf.data.Dataset.from_tensor_slices(({
            "input_ids": input_ids,
            "attention_mask": attention_masks
        }, {
            "start_positions": start_positions,
            "end_positions": end_positions
        }))

    def train(self, contexts, questions, answers, epochs=3, batch_size=8, max_length=256):
        # Prepare training data
        train_dataset = self.prepare_training_data(contexts, questions, answers, max_length).batch(batch_size)

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=optimizer, loss=loss)

        # Train the model
        self.model.fit(train_dataset, epochs=epochs)

    def save_model(self, path):
        # Save the model and tokenizer
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def answer_question(self, context, question):
        # Tokenize and encode the inputs
        inputs = self.tokenizer(question, context, return_tensors="tf", truncation=True)
        
        # Get the model's answer
        outputs = self.model(inputs)
        start_scores, end_scores = outputs.start_logits, outputs.end_logits

        # Find the most probable start and end tokens
        start = tf.argmax(start_scores, axis=1).numpy()[0]
        end = tf.argmax(end_scores, axis=1).numpy()[0]

        # Decode the answer
        answer_tokens = inputs["input_ids"][0][start: end + 1]
        answer = self.tokenizer.decode(answer_tokens)

        return answer
