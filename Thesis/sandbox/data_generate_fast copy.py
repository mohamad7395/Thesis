import os
import transformers
import torch
import pandas as pd
from datasets import Dataset
import random


# Set environment variables
os.environ['HF_TOKEN'] = ""
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ""

# Initialize pipeline
model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map='auto',
    model_kwargs={"torch_dtype": torch.bfloat16},
)

# Define terminators
terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# sample_data = pd.read_csv('/home/s6moakba/Instruct_Restaurants_samples.csv')
sample_data = pd.read_csv('/home/s6moakba/Thesis/sample_train_terms_20k.csv')

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(sample_data)

# Function to generate text for each row
def generate_text(example):
    messages = [{"role": "user", "content": example['prompt']}]
    output = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        temperature=0.6,
        top_p=0.9
    )
    return {'generated_text': output[0]["generated_text"][-1]['content']}


# Apply the function to the entire dataset
dataset = dataset.map(generate_text)

# def generate_text_batch(batch):
#     inputs = batch['prompt']
#     outputs = pipeline(
#         inputs,
#         max_new_tokens=256,
#         eos_token_id=terminators,
#         do_sample=True,
#         pad_token_id=pipeline.tokenizer.eos_token_id,
#         temperature=0.6,
#         top_p=0.9
#     )
    
#     # Directly take the generated text (it should be a string in each element)
#     generated_texts = [output for output in outputs]
    
#     # Return results
#     return {'generated_text': generated_texts}

# # Apply the function in a batched manner
# batch_size = 4  # You can adjust the batch size depending on your GPU's capacity
# dataset = dataset.map(generate_text_batch, batched=True, batch_size=batch_size)

# Convert the dataset back to DataFrame
final_df_with_results = dataset.to_pandas()

# Ensure final DataFrame has the correct columns in the desired order
# final_df_with_results = final_df_with_results[['blank', 'original_terms', 'original_text', 'prompt', 'aspectTerms_old', 'generated_text']]
final_df_with_results = final_df_with_results[['aspect', 'polarity', 'prompt', 'generated_text']]
# Save to CSV
final_df_with_results.to_csv('gen_train_terms_20k.csv', index=False)


class TextGenerator:
    def __init__(self, model_id, csv_path, output_path):
        self.model_id = model_id
        self.csv_path = csv_path
        self.output_path = output_path
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            device_map='auto',
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        self.dataset = Dataset.from_pandas(pd.read_csv(self.csv_path))

    def generate_text(self, example):
        messages = [{"role": "user", "content": example['prompt']}]
        output = self.pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            pad_token_id=self.pipeline.tokenizer.eos_token_id,
            temperature=0.6,
            top_p=0.9
        )
        return {'generated_text': output[0]["generated_text"][-1]['content']}

    def process_dataset(self):
        self.dataset = self.dataset.map(self.generate_text)
        final_df_with_results = self.dataset.to_pandas()
        final_df_with_results = final_df_with_results[['aspect', 'polarity', 'prompt', 'generated_text']]
        final_df_with_results.to_csv(self.output_path, index=False)

# Example usage:
# generator = TextGenerator("meta-llama/Llama-3.1-8B-Instruct", '/home/s6moakba/Thesis/sample_train_terms_20k.csv', 'gen_train_terms_20k.csv')
# generator.process_dataset()