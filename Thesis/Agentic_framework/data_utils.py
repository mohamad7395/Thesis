import pandas as pd
import ast
import random

def prepare_training_terms(csv_path: str):
    """
    Load a dataset and extract aspect terms + polarities for sampling.
    
    Args:
        csv_path (str): Path to the training dataset CSV.
    
    Returns:
        training_terms (set): unique aspect terms
        training_polarities (list): candidate polarities (preset schema)
        df (pd.DataFrame): processed dataset
    """
    df = pd.read_csv(csv_path)

    # Ensure aspectTerms column is parsed from string -> list[dict]
    df['aspectTerms'] = df['aspectTerms'].apply(ast.literal_eval)
    df['aspect'] = df['aspectTerms'].apply(lambda x: [d['term'] for d in x])
    df['polarity'] = df['aspectTerms'].apply(lambda x: [d['polarity'] for d in x])

    training_terms = set(aspect for sublist in df['aspect'] for aspect in sublist)
    training_terms.discard('noaspectterm')

    # Define polarity candidates (could later be learned dynamically)
    training_polarities = ['negative', 'positive', 'neutral']

    return training_terms, training_polarities, df


def get_aspect(training_terms, training_polarities):
    """
    Sample aspect terms + polarities.
    """
    label_len = random.choices([1, 2, 3, 4], weights=[0.6, 0.25, 0.1, 0.05])[0]
    aspec_terms = random.sample(list(training_terms), label_len)
    polarities = random.choices(training_polarities, k=label_len)
    return aspec_terms, polarities


def get_sentences(df: pd.DataFrame, n: int = 3):
    """
    Randomly sample raw_text sentences from a dataframe.
    """
    sampled_rows = df.sample(n=n)
    return sampled_rows['raw_text'].tolist()


def process_terms(terms):
    terms_list = terms[1:-1].split(', ')
    terms_list = [item[1:-1].replace("\\", "") for item in terms_list]
    return terms_list

def create_aspect_terms(aspect,polarity):
    aspect_terms = []
    for aspect, polarity in zip(aspect, polarity):
        aspect_terms.append({'term': aspect, 'polarity': polarity})
    return aspect_terms 


def finalize_output(csv_file:str):
    df = pd.read_csv(csv_file)
    df['Terms'] = df['Terms'].apply(process_terms)
    df['Polarity'] = df['Polarity'].apply(process_terms)
    df['aspectTerms'] = df.apply(lambda x: create_aspect_terms(x['Terms'], x['Polarity']), axis=1)
    df.rename(columns={'sentence':'raw_text'}, inplace=True)
    df['aspectCategories'] = df['aspectTerms'].apply(lambda x: [{'category': 'general', 'polarity': 'neutral'}])
    df['sentenceId'] = df.index
    df.to_csv(csv_file, index=False)
    return df