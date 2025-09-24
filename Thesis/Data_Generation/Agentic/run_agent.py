
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
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama
from evaluate import load
from nltk.corpus import stopwords
from data_utils import prepare_training_terms, get_aspect, get_sentences, process_terms, create_aspect_terms, finalize_output
from agent_tools import AgentTools
from nodes import generator_node, evaluator_node, saver_node, evaluation_decision, save_decision

########################################################################################Constants########################################################################################
MAX_SENTENCES = 2

CSV_FILE = "/home/s6moakba/Thesis/Data_Generation/Agentic/data/TEST.csv"

local_llm = "qwen2.5:14b"

training_terms, training_polarities, train_df = prepare_training_terms(
    "/home/s6moakba/InstructABSA/Dataset/Benchmarks/SemEval14/Train/Restaurants_Train.csv"
)

########################################################################################State########################################################################################

class SentenceState(BaseModel):
    sentence: str = ""  # Stores the generated sentence
    terms: str = ""  # Stores extracted aspect terms
    polarity: str = ""  # Stores polarity values
    is_ok: Literal['OK', 'NOT_OK']  


llm = ChatOllama(model=local_llm, temperature=0.0, base_url="http://localhost:11434")

########################################################################################Tool Functions########################################################################################

agent_tools = AgentTools(training_terms, training_polarities, train_df, llm)

########################################################################################Agents########################################################################################

generator_agent = create_react_agent(
    llm,
    tools=agent_tools.generator_tools(),
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
    tools=agent_tools.evaluator_tools(),
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

########################################################################################Graph########################################################################################
graph = StateGraph(SentenceState)

graph.add_node("generator", lambda s: generator_node(s, generator_agent))
graph.add_node("evaluator", lambda s: evaluator_node(s, evaluator_agent))
graph.add_node("saver", lambda s: saver_node(s, CSV_FILE))

graph.add_edge(START, "generator")
graph.add_edge("generator", "evaluator")
graph.add_conditional_edges("evaluator", evaluation_decision, {"generator": "generator",  "saver": "saver"})
graph.add_conditional_edges("saver", lambda s: save_decision(s, CSV_FILE, MAX_SENTENCES), {"generator": "generator", END: END})

graph = graph.compile()

initial_state = SentenceState(
    sentence="", 
    terms="", 
    polarity="", 
    is_ok="NOT_OK", 
)

########################################################################################Main########################################################################################

if __name__ == "__main__":
    try:
        final_state = graph.invoke(initial_state, {"recursion_limit": 100000})
        finalize_output(CSV_FILE)

    except Exception as e:
        print("An error occurred during graph execution, but we're continuing...")
        print("Error:", e)