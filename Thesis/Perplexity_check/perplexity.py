from evaluate import load
import pandas as pd
import random
import seaborn as sns
import pickle
import matplotlib.pyplot as plt


og_data = pd.read_csv("/home/s6moakba/InstructABSA/Dataset/SemEval16/Train/Restaurants_Train.csv")




val_right =pd.read_csv( "/home/s6moakba/InstructABSA/Dataset/AUG/16/Agentic_16_5k.csv")
val_wrong = pd.read_csv("/home/s6moakba/InstructABSA/Dataset/AUG/16/prompting_16_5k.csv")

val_right = val_right.sample(n=1800, random_state=41)
val_wrong = val_wrong.sample(n=1800, random_state=41)

right_text = val_right['raw_text'].tolist()
wrong_text = val_wrong['raw_text'].tolist()


perplexity = load("perplexity", module_type="metric")


for item, item_name  in zip([right_text, wrong_text], ['agentic', 'promting']):

    results = perplexity.compute(model_id='gpt2',
                                add_start_token=True,
                                predictions=item)
    
    filename = f"/home/s6moakba/Thesis/Perplexity_check/perplexity_results_{item_name}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(results, f)
    
