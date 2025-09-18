import pandas as pd
import transformers
import torch


def pick_random_row(df):
    num_rows = len(df)
    random_index = random.randint(0, num_rows - 1)
    random_row = df.iloc[random_index]
    response = ''
    response+= test_prompt_1
    text = f"\naspect: {random_row['aspect_terms'][1:-1]}\npolarity: {random_row['aspect_polarity'][1:-1]}"
    label = {'aspect_term' : random_row['aspect_terms'],'opinion':random_row['opinion_term'],'polarity':random_row['aspect_polarity'],
             'og_text': random_row['raw_words']}
    response+=text
    response+="\n###Output###(PRINT ONLY THE ANSWER TEXT NO EXPLAINING NOTHING ELSE) "
    return response, label


# df_generated = pd.DataFrame(columns=['label', 'response','generated_text'])
data_generated = []
for i in tqdm(range(1000)):
    new_prompt, new_label = pick_random_row(final_df)
    messages = [{"role": "user", "content":  new_prompt}]
    outputs = pipeline(
        messages,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        pad_token_id=pipeline.tokenizer.eos_token_id,
        temperature=0.6,
        top_p=0.9
    )
    data_generated.append({
        'aspect': new_label['aspect_term'],
        'polarity': new_label['polarity'],
        'prompt': new_prompt,
        'original_text': new_label['og_text'],
        'generated_text': outputs[0]["generated_text"][-1]['content']
    })

############################################################################################################

