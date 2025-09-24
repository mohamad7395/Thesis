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


local_llm = "qwen2.5:14b"


llm = ChatOllama(model=local_llm, temperature=0.0, base_url="http://localhost:11434")



test_mapping= {
    '16': '/home/s6moakba/InstructABSA/Dataset/SemEval16/Test/Restaurants_Test.csv',
    '15': '/home/s6moakba/InstructABSA/Dataset/SemEval15/Test/Restaurants_Test.csv',
    '14_r': '/home/s6moakba/InstructABSA/Dataset/SemEval14/Test/Restaurants_Test.csv',
    '14_l': '/home/s6moakba/InstructABSA/Dataset/SemEval14/Test/Laptops_Test.csv',
}


prompt_mapping = {
    'ate' : """You are Linguistic expert. I want you to find the ASPECT TERM of the review sentence.:
    -Examples: 
    1. After all that, they complained to me about the small tip.
    [noaspectterm] 
    2.Chow fun was dry; pork shu mai was more than usually greasy and had to share a table with loud and rude family. 
    ['Chow fun', 'pork shu mai']
    3. We took advanatage of the half price sushi deal on saturday so it was well worth it.
    ['half price sushi deal']
    4. Decent wine at reasonable prices.
    ['wine']


    Now find the ASPECT TERM of the following sentence and follow ***exactly*** the format of the examples,write nothing else:

    """,

    'atsc' : """You are Linguistic expert. I want you to find the POLARITIES of aspects in review sentence.:
    -Examples: 
    1. After all that, they complained to me about the small tip.
    [none]
    2.Chow fun was dry; pork shu mai was more than usually greasy and had to share a table with loud and rude family. 
    ['negative', 'negative']
    3. We took advanatage of the half price sushi deal on saturday so it was well worth it.
    ['positive']
    4. Decent wine at reasonable prices.
    ['neutral']


    Now find the POLARITIES of aspects in the following sentence and follow ***exactly*** the format of the examples,write nothing else:

    """,

    'aspe' : """You are Linguistic expert. I want you to find the ASPECT TERM and their POLARITIES in review sentence.:
    -Examples: 
    1. After all that, they complained to me about the small tip.
    ['noaspectterm':'none']
    2.Chow fun was dry; pork shu mai was more than usually greasy and had to share a table with loud and rude family. 
    ['Chow fun':'negative', 'pork shu mai':'negative']
    3. We took advanatage of the half price sushi deal on saturday so it was well worth it.
    ['half price sushi deal':'positive']
    4. Decent wine at reasonable prices.
    ['wine':'neutral']


    Now find the  ASPECT TERMS and their POLARITIES in the following sentence and follow ***exactly*** the format of the examples,write nothing else:

    """
}

test_path = "/home/s6moakba/Thesis/agent_practice/qwen_performance/pred/16__ate.csv"

def parse_metrics(test_path):
    """
    Parses the test_path to extract the dataset year and task type.
    Example: '/home/s6moakba/Thesis/agent_practice/qwen_performance/pred/16_ate.csv'
    Returns: ('16', 'ate')
    """
    filename = os.path.basename(test_path)
    parts = filename.split('__')
    if len(parts) >= 2:
        data = parts[0]
        task = parts[1].split('.')[0]
        return {'data': data, 'task': task}
    return None



def automate_training(epoch_addresses):

    for addr in epoch_addresses:
        qwen_pred = []
        print('_'*200)
        print(f"Processing address: {addr}")

        params = parse_metrics(addr)
        data = params['data']
        task = params['task']
        print(f"Data: {data}, Task: {task}")
        id_test_file_path = test_mapping[data]
        task_prompt = prompt_mapping[task]
        id_te_df = pd.read_csv(id_test_file_path)
        for i, row in id_te_df.iterrows():
            text = row['raw_text']
            print(text)
            pred = llm.invoke(task_prompt + text)

            if isinstance(pred, str):
                try:
                    pred = ast.literal_eval(pred)
                except Exception as e:
                    print(f"Error parsing ASPECT_TERM: {e}")
                    pred = []
                    
            qwen_pred.append(pred)
            print(pred)
            print('='*50 )
        id_te_df['qwen_pred'] = qwen_pred
        id_te_df.to_csv(addr, index=False)

if __name__ == "__main__":

    epoch_addresses_aspe = [

        "/home/s6moakba/Thesis/agent_practice/qwen_performance/pred/16__aspe.csv",
        "/home/s6moakba/Thesis/agent_practice/qwen_performance/pred/15__aspe.csv",
        "/home/s6moakba/Thesis/agent_practice/qwen_performance/pred/14_r__aspe.csv",
        "/home/s6moakba/Thesis/agent_practice/qwen_performance/pred/14_l__aspe.csv",
    ]
    automate_training(epoch_addresses_aspe)