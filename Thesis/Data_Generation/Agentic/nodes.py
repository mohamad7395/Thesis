import re
import csv
import os
import pandas as pd
from langchain_core.messages import HumanMessage
from langgraph.graph import START, END

########################################################################################Nodes########################################################################################

def generator_node(state, generator_agent):
    """Node for generating or fixing sentences."""
    print("_"*100)
    print('in generator node')

    result = generator_agent.invoke({"messages": [HumanMessage(content="get the dataset info and generate a sentence with it.")]})

    # print('whole response:', result)
    print('___'*100)

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

def evaluator_node(state, evaluator_agent):
    """Node that evaluates the generated sentence."""
    print("_"*100)
    print('in evaluator node')

    generate_sentence = state.sentence
    gen_terms = state.terms
    gen_polarity = state.polarity
    eval_input_format = f"{generate_sentence} Terms={gen_terms} Polarity={gen_polarity}"
    eval_input = {"messages": [HumanMessage(content=eval_input_format)]}

    result = evaluator_agent.invoke(eval_input)

    eval_response = result["messages"][-1].content
    print('evaluation response: ',eval_response)
    print('___'*100)

    if eval_response not in ["OK", "NOT_OK"]:
        print("[Evaluator Node] Invalid response from LLM. Defaulting to NOT_OK.")
        eval_response = "NOT_OK"

    state.is_ok = eval_response 

    return state


def saver_node(state, CSV_FILE):
    """Node that saves the final approved sentence and its aspect term to a CSV file."""
    print("_"*100)
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


def evaluation_decision(state):
    """Decides whether to loop back to generator or proceed to saver node."""
    print('in sentence evaluation decision')
    
    evaluation_result = state.is_ok

    if evaluation_result == "OK":
        return "saver"

    return "generator"


def review_decision(state):
    """Decides whether to review the generated sentence or proceed to evaluation."""
    print('in review decision')
    review_decision = state.needs_review

    if review_decision == "OK":
        return "saver"

    return "generator"


def save_decision(state, CSV_FILE: str, MAX_SENTENCES: int):
    """Decides whether to continue generating sentences or stop based on saved sentence count."""
    if os.path.isfile(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        sentence_count = len(df)
        if sentence_count < MAX_SENTENCES:
            return "generator"  # Keep generating sentences
    else:
        return "generator"  # If no file exists, continue generating

    return END