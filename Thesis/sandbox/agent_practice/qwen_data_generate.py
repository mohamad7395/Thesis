from typing import Annotated
from typing_extensions import TypedDict
import os
from langchain_ollama import ChatOllama
from typing import Literal
import pandas as pd
import ast
from langgraph.prebuilt import create_react_agent
import re
import csv
from langchain_ollama import ChatOllama
from evaluate import load
import string
import random
from tqdm import tqdm


########################################################################################Constants########################################################################################
MAX_SENTENCES = 500
CSV_FILE_PATH = "/home/s6moakba/Thesis/agent_practice/qwen_generate_14_final_rest.csv"

local_llm = "qwen2.5:14b"
llm = ChatOllama(model=local_llm, temperature=0.0, base_url="http://localhost:11434")


train_set_16 = pd.read_csv('//home/s6moakba/InstructABSA/Dataset/SemEval14/Train/Restaurants_Train.csv')

train_set_16['aspectTerms'] = train_set_16['aspectTerms'].apply(ast.literal_eval)
train_set_16['aspect'] = train_set_16['aspectTerms'].apply(lambda x: [d['term'] for d in x])
train_set_16['polarity'] = train_set_16['aspectTerms'].apply(lambda x: [d['polarity'] for d in x])
training_terms = set(aspect for sublist in train_set_16['aspect'] for aspect in sublist)


training_terms.discard('noaspectterm')


training_polarities =  ['negative', 'positive', 'neutral','negative', 'positive', 'negative', 'positive']

##################################################################################################FUNCTIONS#######################################################################################

def get_aspect():
        label_len = random.choices([1, 2, 3, 4], weights=[0.6, 0.25, 0.1, 0.05])[0]
        aspec_terms = random.sample(list(training_terms),label_len)
        polarities = random.sample(training_polarities,label_len)
        terms = [aspec_terms,polarities]
        return terms

def get_sentences():
    sampled_rows = train_set_16.sample(n=1)
    raw_text = sampled_rows['raw_text'].values[0]
    return raw_text


def process_data_restaurant(length):
    data = []
    for i in range(length):
        aspect_pair = get_aspect()
        terms = aspect_pair[0]
        polarity = aspect_pair[1]
        
        sent = get_sentences
        combined_aspects = ','.join(terms)
        combined_polarities = ','.join(polarity)
        prompt = (

        f"""
                You are a critic who can generate comments on the specified aspect and sentiment\n
                We would like you to complete a sentence generation task. Please follow these requirements:\n
                - You need to use the sentiment,the aspect mentioned in the prompt\n
                - Domain: Restaurants\n
                - Your response must include:
                    1. The sentence.
                    2. A line that starts with `Terms=` followed by the list of aspect terms used.
                    3. A line that starts with `Polarity=` followed by the matching polarity list.
                - ALL aspect terms appear must as actual aspects in the sentence with intended polarities
                - Generated sentence must have the writing style and grammer structure and length of this sentence:\n
                {sent}\n
                - the sentence should not have aspect words that are not specified in the prompt\n
                - DO NOT REPEAT the input text in the output\n"
                - PRINT ONLY THE ANSWER TEXT NO EXPLAINING NOTHING ELSE, MAKE SURE TO USE ASPECT WORDS IN THE OUTPUT\n

                Good Examples : 
                ### input ###
                ['prices'] ['negative']
                ### Output ###
                The prices were too high for this type of restaurant
                Terms: ['prices'] 
                Polarity: ['negative']
                ### input ###
                ['food', 'portions'] ['negative', 'negative']
                ### Output ###
                The food was lousy, too sweet or too salty and the portions tiny
                Terms: ['food', 'portions']
                Polarity: ['negative', 'negative']
                ### input ###
                ['Gnocchi', 'cheesecake'] ['positive', 'negative']
                ### Output ###
                The Gnocchi was perfectly cooked and delicious, but the cheesecake was dry and flavorless.
                Terms= ['Gnocchi', 'cheesecake']
                Polarity= ['positive', 'negative']
                ### input ###
                ['ambience', 'food'] ['positive', 'neurtal']
                ### Output ###
                However, go for the ambience, and consider the food just a companion for a trip across the world!
                Terms= ['ambience', 'food']
                Polarity= ['positive', 'neutral']
                ### input ###
                ['food', 'portions', "Ray's Boathouse"]
                ### Output ###
                The food was lousy - too sweet or too salty and the portions tiny, but Ray's Boathouse had a great view.
                Terms= ['food', 'portions', "Ray's Boathouse"]
                Polarity=  ['negative', 'negative','positive']

                Bad Example : 
                ### input ###
                ['soup'],['positive']
                ### Output ###
                The udon soup was rich and flavorful.
                Terms= ['soup']
                Polarity= ['positive']
                (correct term was soup)
                ### input ###
                ['open kitchen', 'New England Chowder', 'atmosphere'], ['negative', 'negative', 'positive']
                ### Output ###
                Despite the open kitchen adding to the atmosphere, the New England Chowder lacked flavor and freshness.
                Terms= ['open kitchen', 'New England Chowder', 'atmosphere']
                Polarity= ['negative', 'negative', 'positive']
                (open kitchen is not negative)

                Make sure your output **exactly** follows this format. Do not include explanations.\n
                Use plain apostrophes (') in words like "Ray's" or "chef's". Do **not** escape them with backslashes.
                Now complete this task like example with ONE SENTENCE:\n
                ### Input ###\naspect: {combined_aspects}\npolarity: {combined_polarities}\n
                ####Output### 
                """
            )

        data.append({
                'Terms': combined_aspects,
                'Polarity': combined_polarities,
                'prompt': prompt
            })
    return pd.DataFrame(data)


def llm_invoke(prompts):
    return [llm.invoke(prompt) for prompt in prompts]


def extract_raw_text(text):
    text = str(text)
    match = re.search(r'content="(.*?)"\s+additional_kwargs=', text, re.DOTALL)
    if match:
        message_content = match.group(1)
    else:
        message_content = " "

    message_content = message_content.replace("\\n", " ").replace("\n", " ")

    match = re.search(
    r"^(.*?)\.?\s*Terms\s*=\s*(\[[^\]]+\])[,;]?\s*Polarity\s*=\s*(\[[^\]]+\])",
    message_content,
    re.DOTALL)

    if match:
        final_sentence = match.group(1).strip()

    else:
        final_sentence = ''

    return final_sentence



################################################################################################ GENERATE DATA ####################################################################################################

sample_data = process_data_restaurant(MAX_SENTENCES)


########################################################################################Graph########################################################################################

if __name__ == "__main__":

    # sample_data['text'] = llm_invoke(sample_data['prompt'])
    # sample_data['raw_text'] = sample_data['text'].apply(extract_raw_text)
    # for i in range(len(sample_data)):
    #     print(f"________________{i}________________")
    #     print('og text : ', sample_data.iloc[i]['text'])
    #     print("-________________________________")
    #     print('raw text :', sample_data.iloc[i]['raw_text'])



    with tqdm(total=len(sample_data)) as pbar:
        for i, prompt in enumerate(sample_data['prompt']):
            response = llm.invoke(prompt)
            sample_data.at[i, 'text'] = str(response)
            sample_data.at[i, 'sentence'] = extract_raw_text(sample_data.at[i, 'text'])
            pbar.update(1)


    if os.path.exists(CSV_FILE_PATH):
        existing_data = pd.read_csv(CSV_FILE_PATH)
        combined_data = pd.concat([existing_data, sample_data], ignore_index=True)
        combined_data.to_csv(CSV_FILE_PATH, index=False)
    else:
        sample_data.to_csv(CSV_FILE_PATH, index=False)





