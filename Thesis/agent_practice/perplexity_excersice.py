from evaluate import load
import pandas as pd
import random
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

df = pd.read_csv("/home/s6moakba/Thesis/agent_practice/approved_sentences_qwen.csv")

sentence = df['sentence'].iloc[0]

print(sentence)

perplexity = load("perplexity", module_type="metric")

results = perplexity.compute(model_id='gpt2',
                                add_start_token=True,
                                predictions=[sentence])

float(results['mean_perplexity'])
