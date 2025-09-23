import pickle
import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_distances
import nltk
from nltk.stem import WordNetLemmatizer
import re
import time  # For time tracking

# Download necessary NLTK resources (if not already done)
nltk.download('wordnet')
nltk.download('omw-1.4')

# Track the total execution time
start_time = time.time()

# Step 1: Load the pickled sets of nouns and restaurant-related terms
print("Loading noun and restaurant term data...")
with open('/home/s6moakba/Thesis/Filter_nouns/all_nouns_3_4.pkl', 'rb') as f:
    nouns = pickle.load(f)

with open('/home/s6moakba/Thesis/Filter_nouns/instruct_restaurant_nouns.pkl', 'rb') as f:
    restaurant_terms = pickle.load(f)
print(f"Loaded {len(nouns)} nouns and {len(restaurant_terms)} restaurant terms.")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")

# Step 2: Load the pre-trained word2vec model using Gensim's downloader
print("Loading Word2Vec model...")
model_load_start = time.time()
model = api.load("word2vec-google-news-300")
print(f"Model loaded. Time elapsed: {time.time() - model_load_start:.2f} seconds")

# Step 3: Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Step 4: Define a function to filter out words with non-alphabetic characters
def is_valid_word(word):
    return bool(re.fullmatch(r'[a-zA-Z]+', word))

# Step 5: Function to lemmatize and convert a set of words to vectors
def get_lemmatized_vectors(words, model):
    vectors = []
    valid_words = []
    print(f"Processing {len(words)} words...")
    process_start_time = time.time()
    for i, word in enumerate(words):
        if is_valid_word(word) and word in model:
            vectors.append(model[word])
            valid_words.append(word)
        if i % 500 == 0:  # Print progress every 500 words
            print(f"Processed {i}/{len(words)} words.")
    print(f"Finished processing words. Time elapsed: {time.time() - process_start_time:.2f} seconds")
    return np.array(vectors), valid_words

# Step 6: Convert lemmatized restaurant terms and nouns to vectors
print("Converting restaurant terms to vectors...")
restaurant_vectors, valid_restaurant_terms = get_lemmatized_vectors(restaurant_terms, model)
print(f"Converted {len(valid_restaurant_terms)} restaurant terms.")

print("Converting nouns to vectors...")
noun_vectors, valid_nouns = get_lemmatized_vectors(nouns, model)
print(f"Converted {len(valid_nouns)} nouns.")

# Step 7: Compute the centroid of the restaurant-related vectors
print("Computing the centroid of restaurant-related vectors...")
centroid = np.mean(restaurant_vectors, axis=0)
print(f"Centroid computed. Time elapsed: {time.time() - start_time:.2f} seconds")

# Step 8: Compute cosine distances between each noun vector and the restaurant centroid
print("Calculating cosine distances between nouns and the centroid...")
distance_start_time = time.time()
distances_to_centroid = cosine_distances(noun_vectors, centroid.reshape(1, -1))
print(f"Cosine distances calculated. Time elapsed: {time.time() - distance_start_time:.2f} seconds")

# Step 9: Filter nouns based on a similarity threshold (e.g., 0.5)
threshold = 0.4
print(f"Filtering nouns with distance threshold: {threshold}")
restaurant_nouns = {valid_nouns[i] for i in range(len(valid_nouns)) if distances_to_centroid[i][0] < threshold}
print(f"Filtered {len(restaurant_nouns)} nouns related to the restaurant domain.")

# Step 10: Print or save the filtered nouns related to the restaurant domain
print("Final filtered restaurant-related nouns:")
print(restaurant_nouns)

# Optionally, save the result to a file
with open('restaurant_related_nouns_3_4.pkl', 'wb') as f:
    pickle.dump(restaurant_nouns, f)

# Final execution time
print(f"Total execution time: {time.time() - start_time:.2f} seconds")
