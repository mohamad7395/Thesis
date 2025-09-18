import os
import torch
import transformers
from langchain_community.llms import HuggingFacePipeline
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Set Hugging Face API Token for local model usage
os.environ['HF_TOKEN'] = ""
os.environ['HUGGINGFACEHUB_API_TOKEN'] = ""

# Initialize local pipeline
model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200, eos_token_id=tokenizer.eos_token_id)

llm = HuggingFacePipeline(pipeline=pipe)

# Simple function for text modification
def simple_paraphrase(text):
    response = llm(f"Paraphrase the following sentence: {text}\n\nOnly return the paraphrased sentence, nothing else.")
    return response.strip().split("\n")[0]

# Define a single tool for simplicity
paraphrase_tool = Tool(
    name="ParaphraseTool",
    func=simple_paraphrase,
    description="Use this tool to paraphrase a given text."
)

# Initialize a basic LangChain agent
agent = initialize_agent(
    tools=[paraphrase_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,  # Prevents crashing on parsing errors
    max_iterations=1  
)

# Test the agent
if __name__ == "__main__":
    query = "Paraphrase this sentence only 1 time: The quick brown fox jumps over the lazy dog."
    print(agent.run(query))