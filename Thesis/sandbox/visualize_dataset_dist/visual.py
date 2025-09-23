import pandas as pd 
from collections import Counter
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import Counter
import ast


def load_df(file_path):
    df = pd.read_csv(file_path)
    df['aspectTerms'] = df['aspectTerms'].apply(ast.literal_eval)
    df['aspect'] = df['aspectTerms'].apply(lambda x: [d['term'] for d in x])
    return df
def ge_counter(df):
    aspect_counter = Counter([aspect for sublist in df['aspect'] for aspect in sublist])
    return aspect_counter

def get_embedding(text, pooling='mean', max_length=512):
    """
    Generate an embedding for the input text by passing it through the T5 encoder.
    """

    model_checkpoint = 'allenai/tk-instruct-base-def-pos'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    encoder_outputs = model.encoder(**inputs)
    last_hidden_state = encoder_outputs.last_hidden_state  # shape: (batch_size, seq_length, hidden_size)
    
    if pooling == 'mean':
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = torch.sum(last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
        else:
            pooled = last_hidden_state.mean(dim=1)
    elif pooling == 'first':
        pooled = last_hidden_state[:, 0, :]
    else:
        raise ValueError("Unsupported pooling type. Choose 'mean' or 'first'.")
    
    return pooled.detach().cpu().numpy()[0]


def main():
    file_path_1 = '/home/s6moakba/InstructABSA/Dataset/gen_train_15_10k.csv'
    df_1 = load_df(file_path_1)

    file_path_2 = '/home/s6moakba/InstructABSA/Dataset/SemEval15/Train/Restaurants_Train.csv'
    df_2 = load_df(file_path_2)

    df = pd.concat([df_1, df_2], ignore_index=True, axis=0)

    aspect_counter = ge_counter(df)
    unique_embeddings = []
    
    unique_phrases = list(aspect_counter.keys())

    threshold = np.percentile(list(aspect_counter.values()), 90)

    
    for aspect in unique_phrases:
        unique_embeddings.append(get_embedding(aspect))

    unique_embeddings = np.array(unique_embeddings)

    tsne = TSNE(n_components=2, perplexity=3, random_state=42)
    embeddings_2d = tsne.fit_transform(unique_embeddings)

    plt.figure(figsize=(20, 16))

# We use the frequency (count) to determine the size of each marker.
# Adjust the scaling factor as needed.
    sizes = [aspect_counter[phrase] * 100 for phrase in unique_phrases]

    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                        s=sizes, c='cornflowerblue', alpha=0.7, edgecolor='k')

    # Annotate each point with its phrase and frequency
    for i, phrase in enumerate(unique_phrases):
        if aspect_counter[phrase] >= threshold:
            plt.annotate(f"{phrase} ({aspect_counter[phrase]})",
                        (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                        fontsize=12, xytext=(5, 2), textcoords='offset points')

    plt.title("t-SNE Visualization with Frequency-Based Marker Sizes")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.savefig('/home/s6moakba/Thesis/visualize_dataset_dist/15_both.png')
    print('Done')
    # plt.show()

    


if __name__ == "__main__":
    main()