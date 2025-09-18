
import pandas as pd 
import numpy as np
import pickle
import random
import ast

def make_dict_from_df(df):
    """
    Make a dictionary from the dataframe
    """
    df['aspectTerms'] = df['aspectTerms'].apply(ast.literal_eval)
    df['aspect'] = df['aspectTerms'].apply(lambda x: [d['term'] for d in x])
    training_terms = set(aspect for sublist in df['aspect'] for aspect in sublist)
    return training_terms

def process_data(noun_set,row_num, aspect_len):
    data = []
    polarity_options = ['Positive', 'Neutral', 'Negative','Negative','Positive','Negative','Positive' ]
    for i in range(row_num):
        terms = random.sample(list(noun_set), aspect_len)
        polarity = random.sample(polarity_options, aspect_len)
        combined_aspects = ','.join(terms)
        combined_polarities = ','.join(polarity)
        prompt = (
                f"You are a critic who can generate comments on the specified aspect and sentiment\n"
                f"We would like you to complete a sentence generation task. Please follow these requirements:\n"
                f"- You need to use the sentiment,the aspect mentioned in the prompt\n"
                f"- Domain: Restaurants\n"
                f"- The generated sentence must be in length within 100 words.\n"
                f"- the sentence should not have aspect words that are not specified in the prompt\n"
                f"- DO NOT REPEAT the input text in the output\n"
                f"- PRINT ONLY THE ANSWER TEXT NO EXPLAINING NOTHING ELSE, MAKE SURE TO USE ASPECT WORDS IN THE OUTPUT\n"
                f"- examples:\n"
                f"### input ###\n"
                f"aspect: prices\n"
                f"polarity: negative\n"
                f"### Output ###\n"
                f"The prices were too high for this type of restaurant\n"
                f"### input ###\n"
                f"aspect: vibe, owner, service\n"
                f"polarity: positive,positive,negative\n"
                f"### Output ###\n"
                f"Best of all is the warm vibe , the owner is super friendly but service isn't fast .\n"
                f"### input ###\n"
                f"aspect: bar, table, dinner\n"
                f"polarity: Positive, Neutral, Neutral\n"
                f"### Output ###\n"
                f"After really enjoying ourselves at the bar we sat down at a table and had dinner\n"
                f"Now complete this task like example with ONE SENTENCE:\n"
                f"### Input ###\naspect: {combined_aspects}\npolarity: {combined_polarities}\n###Output### "
            )
        data.append({
                'aspect': combined_aspects,
                'polarity': combined_polarities,
                'prompt': prompt
            })
    return pd.DataFrame(data)
        
def sample_data(nouns, sample_list):
    """
    Sample the data
    """
    sample_data_1_aspect = process_data(nouns, sample_list[0], 1)
    sample_data_2_aspect = process_data(nouns, sample_list[1], 2)
    sample_data_3_aspect = process_data(nouns, sample_list[2], 3)
    sample_data_4_aspect = process_data(nouns, sample_list[3], 4)

    smaple_data = pd.concat([sample_data_1_aspect, sample_data_2_aspect, sample_data_3_aspect, sample_data_4_aspect])
    return smaple_data 

     