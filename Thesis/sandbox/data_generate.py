import os
import transformers
import torch
import pandas as pd

# Set the Hugging Face token
os.environ['HF_TOKEN'] = ""
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ""


model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    device_map='auto',
    model_kwargs={"torch_dtype": torch.bfloat16},
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


final_df = pd.read_csv('Instruct_Restaurants.csv')
# Read the first 1/3 of the final_df
final_df = final_df.iloc[:len(final_df)//4]

test_prompt_1 = """You are a critic who can generate comments on the specified aspect and sentiment
We would like you to complete a sentence generation task, and we will tell you how to generate appropriate sentences. Please follow these requirements:
- You need to use the sentiment, opinion, and the aspect mentioned in the prompt
- Domain of sample generation: Restaurants
- Generate a sentence containing a given aspect, clarify the meaning of the aspect, and
generate sentences corresponding to the polarity of the sentiment.
- The generated sentence must be in length within 100 words.
- Generated sentences can contain only one period at a time and the sentence should not
consist of an unspecified aspect 
- DO NOT REPEAT the input text in the output
- examples:
### input ###
aspect: prices
polarity: negative
### Output ### 
The prices were too high for this type of restaurant
### input ###
aspect: vibe, owner, service
polarity: positive,positive,positive
### Output ### 
"Best of all is the warm vibe , the owner is super friendly and service is fast ."

Now complete this task like example with ONE SENTENCE:
### Input ###"""
test_prompt_2 = """aspect: menu
opinion: diverse
polarity: positive
Output:"""

import pandas as pd
from tqdm import tqdm
import itertools
import random

def process_row(row, polarity_combination,combined_aspects):
    response = ''
    response += test_prompt_1
    
    # Combine the aspect terms and their corresponding polarities into a single string
    combined_aspects = ','.join(combined_aspects)
    combined_polarities = ','.join(polarity_combination)
    
    text = f"\naspect: {combined_aspects}\npolarity: {combined_polarities}"
    label = {
        'aspect_term': combined_aspects,
        'polarity': combined_polarities,
        'og_text': row['raw_words']
    }
    
    response += text
    response += "\n###Output###(PRINT ONLY THE ANSWER TEXT NO EXPLAINING NOTHING ELSE) "
    return response, label

# Create a list to hold the generated data
data_generated = []

# Iterate through each row in the DataFrame using tqdm for progress bar
for _, row in tqdm(final_df.iterrows(), total=len(final_df)):
    # Extract aspect terms (removing square brackets and splitting by commas)
    combined_aspect_terms = row['aspect_terms'][1:-1].replace("'", "").split(", ")
    
    # Generate all combinations of polarities matching the number of aspect terms
    polarity_options = ['Positive', 'Neutral', 'Negative']
    polarity_combinations = list(itertools.product(polarity_options, repeat=len(combined_aspect_terms)))
    for polarity_combination in polarity_combinations:
        # Generate the prompt and label
        new_prompt, new_label = process_row(row, polarity_combination,combined_aspect_terms)

        # Generate output using the pipeline
        messages = [{"role": "user", "content": new_prompt}]
        outputs = pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            pad_token_id=pipeline.tokenizer.eos_token_id,
            temperature=0.6,
            top_p=0.9
        )
        
        # Store the generated data
        data_generated.append({
            'aspect': new_label['aspect_term'],
            'polarity': new_label['polarity'],
            'prompt': new_prompt,
            'original_text': new_label['og_text'],
            'generated_text': outputs[0]["generated_text"][-1]['content']
        })

# Optionally convert the data_generated list to a DataFrame
df_generated = pd.DataFrame(data_generated)


df_generated = pd.DataFrame(data_generated)

df_generated.to_csv('gen+4.csv', index=False)