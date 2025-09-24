import os
import pandas as pd
from tqdm import tqdm
from langchain_ollama import ChatOllama
from prompting_utils import prepare_training_terms, process_data_restaurant, extract_raw_text, postprocess_generated_data

#########################################################################################Constants########################################################################################
MAX_SENTENCES = 2
CSV_FILE_PATH = "/home/s6moakba/Thesis/Data_Generation/Prompting/data/test.csv"
TRAIN_PATH = "/home/s6moakba/InstructABSA/Dataset/Benchmarks/SemEval14/Train/Restaurants_Train.csv"
local_llm = "qwen2.5:14b"

llm = ChatOllama(model=local_llm, temperature=0.0, base_url="http://localhost:11434")

#########################################################################################Data Preparation########################################################################################
training_terms, training_polarities, train_df = prepare_training_terms(TRAIN_PATH)
sample_data = process_data_restaurant(MAX_SENTENCES, training_terms, training_polarities, train_df)

###########################################################################################Text Generation########################################################################################

if __name__ == "__main__":
    with tqdm(total=len(sample_data)) as pbar:
        for i, prompt in enumerate(sample_data['prompt']):
            response = llm.invoke(prompt)
            sample_data.at[i, 'text'] = str(response)
            sample_data.at[i, 'sentence'] = extract_raw_text(response)
            pbar.update(1)
    processed_data = postprocess_generated_data(sample_data)

    if os.path.exists(CSV_FILE_PATH):
        existing = pd.read_csv(CSV_FILE_PATH)
        combined = pd.concat([existing, processed_data], ignore_index=True)
        combined.to_csv(CSV_FILE_PATH, index=False)
    else:
        processed_data.to_csv(CSV_FILE_PATH, index=False)
