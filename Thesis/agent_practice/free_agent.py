from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
import os
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.messages import ToolMessage
from typing import Literal
import pandas as pd
import ast
from langgraph.prebuilt import create_react_agent
import re
import csv
from langchain_ollama import ChatOllama
from evaluate import load
import nltk
from nltk.corpus import stopwords
import string



########################################################################################Constants########################################################################################
MAX_SENTENCES = 7000

CSV_FILE = "/home/s6moakba/Thesis/agent_practice/approved_qwen_14_r.csv"

local_llm = "qwen2.5:14b"

# llama = "llama3.1:8b"

train_set_16 = pd.read_csv('/home/s6moakba/InstructABSA/Dataset/SemEval14/Train/Restaurants_Train.csv')


train_set_16['aspectTerms'] = train_set_16['aspectTerms'].apply(ast.literal_eval)
train_set_16['aspect'] = train_set_16['aspectTerms'].apply(lambda x: [d['term'] for d in x])
train_set_16['polarity'] = train_set_16['aspectTerms'].apply(lambda x: [d['polarity'] for d in x])
training_terms = set(aspect for sublist in train_set_16['aspect'] for aspect in sublist)


training_terms.discard('noaspectterm')


training_polarities =  ['negative', 'positive', 'neutral','negative', 'positive', 'negative', 'positive']

########################################################################################State########################################################################################

class SentenceState(BaseModel):
    sentence: str = ""  # Stores the generated sentence
    terms: str = ""  # Stores extracted aspect terms
    polarity: str = ""  # Stores polarity values
    is_ok: Literal['OK', 'NOT_OK']  
    needs_review: Literal['OK', 'NOT_OK']    # Whether the sentence requires review


llm = ChatOllama(model=local_llm, temperature=0.0, base_url="http://localhost:11434")

########################################################################################Functions########################################################################################

import random
def get_aspect():
        # label_len = random.randint(1, 1)
        label_len = random.choices([1, 2, 3, 4], weights=[0.6, 0.25, 0.1, 0.05])[0]
        aspec_terms = random.sample(list(training_terms),label_len)
        polarities = random.sample(training_polarities,label_len)
        terms = [aspec_terms,polarities]
        return terms


def get_sentences():
    sampled_rows = train_set_16.sample(n=3)
    raw_text = sampled_rows['raw_text'].tolist()
    return raw_text



########################################################################################Tool Functions########################################################################################
@tool
def get_info() -> str:
    """Extracts the information from sample sentences."""
    print("_"*200)
    print('in get info tool')


    samples = get_sentences()
    print('Samples:', samples)
    prompt = f"""Analyze the following sentences and identify:
    1. The dominant writing style 
    2. The dominant grammar structure 
    3. The dominant length of the sentences (short = less than 10 words, medium = between 10 15 words, long = more than 15 words)

    Output format (JSON):  
        {{"writing_style": "...", "grammar_structure": "...", "length": "..."}}  

    Use ***one*** JSON that describes all of the sentences. Do not analyze each sentence individually.
    Make sure your output **exactly** follows this format. Do not include explanations.
    Sentences:\n""" + "\n".join(f'- "{s}"' for s in samples)

    response = llm.invoke(prompt)
    print('Response:', response.content)

    print('Info extracted')
    print('_'*100)

    return response

@tool
def generate_sentences(style_info: dict) -> str:
    """
    Generates a sentence using aspect terms and a given writing style.
    Arguments:
    style_info: dictionary containing keys:
                - "writing_style"
                - "grammar_structure"
                - "length"
                """
    print("_"*200)
    print('Generating sentence')
    
    terms = get_aspect()
    aspect_term = terms[0]
    polarity = terms[1]

    writing_style = style_info.get("writing_style", "unknown")
    grammar_structure = style_info.get("grammar_structure", "unknown")
    sentence_length = style_info.get("length", "unknown")

    print('Selected terms:', aspect_term)
    print('Selected polarities:', polarity)



    prompt = f"""
    You are a critic who can generate comments on the specified aspect and sentiment
    We would like you to complete a sentence generation task. Please follow these requirements:

    ###TASK###
    - Generate a sentence using this aspect term: {', '.join(aspect_term)} with the following polarities : {', '.join(polarity)}.
    Write in the style: {writing_style}, and use a {grammar_structure} grammatical structure and {sentence_length} sentence length.
    
    ###REQUIREMENTS###
    - Your response must include:
    1. The sentence.
    2. A line that starts with `Terms=` followed by the list of aspect terms used.
    3. A line that starts with `Polarity=` followed by the matching polarity list.
    - Domain: Restaurants
    - the sentence should not have aspect words that are not specified in the prompt
    - Use the exact structure shown in the examples below.

    Good Examples : 
    ### input ###
    ['prices'] ['negative']
    ### Output ###
    The prices were too high for this type of restaurant
    Terms: ['prices'] 
    Polarity: ['negative']
    ### input ###
    ['ambience', 'food'] ['positive', 'neurtal']
    ### Output ###
    However, go for the ambience, and consider the food just a companion for a trip across the world!
    Terms= ['ambience', 'food']
    Polarity= ['positive', 'neutral']
    ### input ###
    ['food', 'portions', "Ray's Boathouse"]
    ### Output ###
    sentence= The food was lousy - too sweet or too salty and the portions tiny, but Ray's Boathouse had a great view.
    Terms= ['food', 'portions', "Ray's Boathouse"]
    Polarity=  ['negative', 'negative','positive']

    Bad Example : 
    ### input ###
    ['soup'],['positive']
    ### Output ###
    sentence= The udon soup was rich and flavorful.
    Terms= ['soup']
    Polarity= ['positive']
    (correct term was soup)


    Make sure your output **EXACTLY** follows this format. Do not include explanations.
    Use plain apostrophes (') in words like "Ray's" or "chef's". Do **not** escape them with backslashes.
"""
    response = llm.invoke(prompt)
    print('Sentence generated')
    return response


@tool
def evaluate_sentence(text: str) -> str:
    """
    Evaluates if the provided *aspect terms* and their corresponding *polarities* are *correctly* used in the sentence
    Responds only with 'OK' or 'NOT_OK'.
    """
    print("_"*200)
    print('Evaluating sentence')
    prompt = f"""
    You are an expert in linguistic evaluation. Your task is to check if the given aspect terms and polarities are correct for the provided sentence.
    
    - If ALL aspect terms appear as actual aspects in the sentence with intended polarities, respond **only** with: OK
    - If any term is missing, incorrect, or not an aspect of the sentence,or wrong polarities respond **only** with: NOT_OK

    Do not provide explanations or any other text.

    Example Input:
    The food was lousy, too sweet or too salty and the portions tiny. Terms= ['food', 'portions'], Polarity= ['negative', 'negative']
    OK
    The Gnocchi was perfectly cooked and delicious, but the cheesecake was dry and flavorless. Terms=['Gnocchi', 'cheesecake'], Polarity= ['positive', 'negative']
    OK

    Bad Example:
    The udon soup was rich and flavorful. Terms= ['soup'], Polarity= ['positive']
    NOT_OK
    Despite the open kitchen adding to the atmosphere, the New England Chowder lacked flavor and freshness. Terms= ['open kitchen', 'New England Chowder', 'atmosphere'], Polarity= ['negative', 'negative', 'positive']
    NOT_OK

    Make sure your output **exactly** follows this format. Do not include explanations.
    Input:
    {text}
    """
    return llm.invoke(prompt)


@tool
def label_inclusion(text: str) -> str:
    """
    Checks if all aspect terms are present in the sentence.
    Input format: <sentence> Terms=[...] Polarity=[...]
    Returns 'OK' if all terms are present in the sentence, otherwise 'NOT_OK'.
    """
    print("_"*200)
    print('in label inclusion tool')

    try:
        # Extract sentence, terms, and polarity using regex
        sentence_match = re.match(r'^(.*?)\s+Terms=', text)
        terms_match = re.search(r'Terms=\[(.*?)\]', text)

        if not sentence_match or not terms_match:
            return 'NOT_OK'

        sentence = sentence_match.group(1).strip()
        terms_str = terms_match.group(1)
        terms = [term.strip().strip("'\"") for term in terms_str.split(',') if term.strip()]
        print('terms:',terms)
        print('sentence:',sentence)
        # Check if every term appears in the sentence
        for term in terms:
            if term not in sentence:
                return 'NOT_OK'
        return 'OK'

    except Exception as e:
        print(f"[label_inclusion] Error: {e}")
        return 'NOT_OK'






########################################################################################Agents########################################################################################

generator_agent = create_react_agent(
    llm,
    tools=[generate_sentences, get_info],
    prompt="""
            You are a sentence generator.
            Your job is to:
            1. Call the `get_info` tool to understand the dominant writing style, and grammar structure, and senetnce length in the dataset.
            2. Then, call the `generate_sentences` tool using the `style_info` you got from `get_info`.

        ***USE EACH TOOL ONLY ONCE.***
"""
)


evaluator_agent = create_react_agent(
    llm,
    tools=[label_inclusion,evaluate_sentence],
    prompt=""""
            You are an evaluator.

            Your job is to:
            1. Call the `label_inclusion` tool to check if all aspect terms are present in the sentence.
            2. If the result is 'OK', call the `evaluate_sentence` tool to verify the aspect-polarity correctness.

            Respond with 'OK' only if both tools return OK. Otherwise, respond with 'NOT_OK'.

            ***USE EACH TOOL ONLY ONCE.***
            ***DO NOT SKIP STEPS OR GUESS.***
"""


)


########################################################################################Nodes########################################################################################
def generator_node(state: SentenceState):
    """Node for generating or fixing sentences."""
    print("_"*200)
    print('in generator node')

    result = generator_agent.invoke({"messages": [HumanMessage(content="get the dataset info and generate a sentence with it.")]})

    # print('whole response:', result)
    print('___'*200)

    message_content = result["messages"][-2].content.strip().__str__() 

    match = re.search(r'content="(.*?)"\s+additional_kwargs=', message_content, re.DOTALL)
    if match:
        message_content = match.group(1)
    else:
        message_content = message_content

    print('generator node result:',message_content)

    message_content = message_content.replace("\\n", " ").replace("\n", " ")

    match = re.search(
    r"^(.*?)\.?\s*Terms\s*=\s*(\[[^\]]+\])[,;]?\s*Polarity\s*=\s*(\[[^\]]+\])",
    message_content,
    re.DOTALL)

    if match:
        final_sentence = match.group(1).strip()
        aspect_term = match.group(2).strip()
        polarity = match.group(3).strip()
    else:
        final_sentence = ''
        aspect_term = "Unknown Terms"
        polarity = "Unknown Polarity"

    final_sentence = final_sentence.removeprefix("sentence=").strip()

    # Update the state
    state.sentence = final_sentence
    state.terms = aspect_term
    state.polarity = polarity

    return state

def evaluator_node(state: SentenceState):
    """Node that evaluates the generated sentence."""
    print("_"*200)
    print('in evaluator node')

    generate_sentence = state.sentence
    gen_terms = state.terms
    gen_polarity = state.polarity
    eval_input_format = f"{generate_sentence} Terms={gen_terms} Polarity={gen_polarity}"
    eval_input = {"messages": [HumanMessage(content=eval_input_format)]}

    result = evaluator_agent.invoke(eval_input)

    eval_response = result["messages"][-1].content
    print('evaluation response: ',eval_response)
    print('___'*200)

    if eval_response not in ["OK", "NOT_OK"]:
        print("[Evaluator Node] Invalid response from LLM. Defaulting to NOT_OK.")
        eval_response = "NOT_OK"

    state.is_ok = eval_response 

    return state


def review_node(state: SentenceState):
    """Node to review the generated sentence."""
    print("_"*100)
    print('in review node')

    final_sentence = state.sentence
    aspect_term = state.terms
    polarity = state.polarity

    review_result = label_inclusion(final_sentence, aspect_term)
    state.needs_review = review_result

    return state

def saver_node(state: SentenceState):
    """Node that saves the final approved sentence and its aspect term to a CSV file."""
    print("_"*200)
    print('in saver node')
    
    final_sentence = state.sentence
    aspect_term = state.terms
    polarity = state.polarity

    csv_file = CSV_FILE

    # Ensure the file has headers if it does not exist
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["sentence", "Terms", "Polarity"])  # Write headers if new file
        writer.writerow([final_sentence, aspect_term, polarity])

    print(f"[Saver Node] Sentence saved: {final_sentence} | Aspect Term: {aspect_term}")

    state.sentence = ''
    state.terms = ''
    state.polarity = ''

    return state

########################################################################################Decision########################################################################################


def evaluation_decision(state: SentenceState):
    """Decides whether to loop back to generator or proceed to saver node."""
    print('in sentence evaluation decision')
    
    evaluation_result = state.is_ok

    if evaluation_result == "OK":
        return "saver"

    return "generator"


def review_decision(state: SentenceState):
    """Decides whether to review the generated sentence or proceed to evaluation."""
    print('in review decision')
    review_decision = state.needs_review

    if review_decision == "OK":
        return "saver"

    return "generator"


def save_decision(state: SentenceState):
    """Decides whether to continue generating sentences or stop based on saved sentence count."""
    if os.path.isfile(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        sentence_count = len(df)
        if sentence_count < MAX_SENTENCES:
            return "generator"  # Keep generating sentences
    else:
        return "generator"  # If no file exists, continue generating

    return END


########################################################################################Graph########################################################################################
graph = StateGraph(SentenceState)

graph.add_node("generator", generator_node)
graph.add_node("evaluator", evaluator_node)
# graph.add_node("reviewer", review_node)
graph.add_node("saver", saver_node)


graph.add_edge(START, "generator")
graph.add_edge("generator", "evaluator")
graph.add_conditional_edges("evaluator", evaluation_decision, {"generator": "generator",  "saver": "saver"})

# graph.add_edge("generator", "reviewer")
# graph.add_conditional_edges("reviewer", review_decision, {"generator": "generator", "saver": "saver"})
graph.add_conditional_edges("saver", save_decision, {"generator": "generator", END: END})

# graph.add_node("generator", generator_node)
# graph.add_edge(START, "generator")
# graph.add_edge("generator", END)



graph = graph.compile()


initial_state = SentenceState(
    sentence="", 
    terms="", 
    polarity="", 
    is_ok="NOT_OK", 
    needs_review="NOT_OK"
)



if __name__ == "__main__":
    try:
        final_state = graph.invoke(initial_state, {"recursion_limit": 100000})
    except Exception as e:
        print("An error occurred during graph execution, but we're continuing...")
        print("Error:", e)