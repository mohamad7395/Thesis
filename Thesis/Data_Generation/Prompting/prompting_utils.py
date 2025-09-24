import random, re, ast, os
import pandas as pd
from tqdm import tqdm

def prepare_training_terms(dataset_path: str):
    df = pd.read_csv(dataset_path)
    df['aspectTerms'] = df['aspectTerms'].apply(ast.literal_eval)
    df['aspect'] = df['aspectTerms'].apply(lambda x: [d['term'] for d in x])
    df['polarity'] = df['aspectTerms'].apply(lambda x: [d['polarity'] for d in x])

    training_terms = {a for sub in df['aspect'] for a in sub}
    training_terms.discard('noaspectterm')
    training_polarities = ['negative', 'positive', 'neutral',
                           'negative', 'positive',
                           'negative', 'positive']
    return training_terms, training_polarities, df

def get_aspect(training_terms, training_polarities):
    label_len = random.choices([1, 2, 3, 4], weights=[0.6, 0.25, 0.1, 0.05])[0]
    aspect_terms = random.sample(list(training_terms), label_len)
    polarities = random.sample(training_polarities, label_len)
    return aspect_terms, polarities

def get_sentence(train_df):
    return train_df.sample(n=1)['raw_text'].values[0]

def process_data_restaurant(length, training_terms, training_polarities, train_df):
    data = []
    for i in range(length):
        aspect_pair = get_aspect(training_terms, training_polarities)
        terms = aspect_pair[0]
        polarity = aspect_pair[1]
        
        sent = get_sentence(train_df)
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

import pandas as pd

def create_aspect_terms(aspects, polarities):
    """
    Combine aspect terms and polarities into a list of dicts.
    """
    return [{'term': a, 'polarity': p} for a, p in zip(aspects, polarities)]



def postprocess_generated_data(df_or_path):
    """
    Postprocess generated data into ABSA format.
    Accepts a DataFrame (preferred for in-memory) or CSV path.
    Returns a cleaned DataFrame.
    """
    if isinstance(df_or_path, str):
        gen_data = pd.read_csv(df_or_path)
    else:
        gen_data = df_or_path.copy()

    # Split comma-joined fields
    gen_data['Terms'] = gen_data['Terms'].apply(lambda x: x.split(','))
    gen_data['Polarity'] = gen_data['Polarity'].apply(lambda x: x.split(','))

    # Handle missing sentences
    gen_data['sentence'] = gen_data['sentence'].apply(lambda x: x if isinstance(x, str) else '.')

    # Build aspectTerms
    gen_data['aspectTerms'] = gen_data.apply(
        lambda x: create_aspect_terms(x['Terms'], x['Polarity']),
        axis=1
    )

    # Rename + add extra fields
    gen_data.rename(columns={'sentence': 'raw_text'}, inplace=True)
    gen_data['aspectCategories'] = gen_data['aspectTerms'].apply(
        lambda _: [{'category': 'general', 'polarity': 'neutral'}]
    )
    gen_data.drop(columns=['text', 'prompt'], inplace=True, errors='ignore')
    gen_data['sentenceId'] = gen_data.index

    return gen_data