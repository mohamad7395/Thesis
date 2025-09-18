
print('_______importing_______')
import sys
import os

import torch

# Add the root path to sys.path
# Add the root path to sys.path
root_path = '/home/s6moakba/InstructABSA'
sys.path.append(root_path)
os.chdir(root_path)
    
os.chdir(root_path)
use_mps = True if torch.has_mps else False

from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator
from instructions import InstructionsHandler

import pandas as pd


# model_out_path = '/home/s6moakba/InstructABSA/Models_baseline/ate/allenaitk-instruct-base-def-pos-lapt2014_iabsa1'
# /home/s6moakba/InstructABSA/Models_baseline
model_out_path = '/home/s6moakba/InstructABSA/Models_baseline_ASPE_16'


# /home/s6moakba/InstructABSA/baseline_16/ate/allenaitk-instruct-base-def-pos-lapt2014_iabsa1
# Load the data

id_train_file_path = './Dataset/SemEval16/Train/Restaurants_Train.csv'

id_train_generated_file_path = '/home/s6moakba/InstructABSA/Dataset/AUG/16/Agentic_16_5k.csv'

id_test_file_path = './Dataset/SemEval16/Test/Restaurants_Test.csv'


id_tr_df = pd.read_csv(id_train_file_path)


# validation_ratio = 0.1
# num_samples = len(id_tr_df)
# num_validation_samples = int(validation_ratio * num_samples)

# id_val_df = id_tr_df.sample(n=num_validation_samples, random_state=42)
# id_tr_df = id_tr_df.drop(id_val_df.index)


# id_tr_df_gen = pd.read_csv(id_train_generated_file_path)

id_val_df = pd.read_csv(id_train_generated_file_path)
id_val_df = id_val_df.sample(n=1800, random_state=41)



# id_tr_df = pd.concat([id_tr_df, id_tr_df_gen], ignore_index=True, axis=0)

id_te_df = pd.read_csv(id_test_file_path)
print("_"*50)

# Use id_tr_df for training and id_val_df for validation

# Get the input text into the required format using Instructions
# Get the input text into the required format using Instructions
instruct_handler = InstructionsHandler()

instruct_handler.load_instruction_set1()

# Set instruction_set1 for InstructABSA-1 and instruction_set2 for InstructABSA-2
loader = DatasetLoader(train_df_id = id_tr_df, test_df_id = id_te_df, val_df_id =id_val_df )
# Set bos_instruct1 for lapt14 and bos_instruct2 for rest14. For other datasets, modify the insructions.py file.
if loader.train_df_id is not None:
    loader.train_df_id = loader.create_data_in_aspe_format(loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', 
                                                                 instruct_handler.aspe['bos_instruct2'], instruct_handler.aspe['eos_instruct'])
if loader.test_df_id is not None:
    loader.test_df_id = loader.create_data_in_aspe_format(loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', 
                                                                instruct_handler.aspe['bos_instruct2'], instruct_handler.aspe['eos_instruct'])
if loader.val_df_id is not None:
    loader.val_df_id = loader.create_data_in_aspe_format(loader.val_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', 
                                                                instruct_handler.aspe['bos_instruct2'], instruct_handler.aspe['eos_instruct'])



print('_________model_inference_________')

# Model inference - Loading from Checkpoint
t5_exp = T5Generator(model_out_path)

# Tokenize Datasets
id_ds, id_tokenized_ds, ood_ds, ood_tokenzed_ds = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs)

# Get prediction labels - Training set   
# id_tr_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'train', batch_size = 16)
# id_tr_labels = [i.strip() for i in id_ds['train']['labels']]

# # Get prediction labels - Testing set
# id_te_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'test',  batch_size = 16)
# id_te_labels = [i.strip() for i in id_ds['test']['labels']]

id_val_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'validation',  batch_size = 16)
id_val_labels = [i.strip() for i in id_ds['validation']['labels']]



# Save id_te_labels and id_te_pred_labels to a file
with open('./Label_compare/agentic_16_aspe.txt', 'w') as f:
    f.write('id_gen_labels: ' + str(id_val_labels) + '\n')
    f.write('id_gen_pred_labels: ' + str(id_val_pred_labels) + '\n')

# with open('./Label_compare/baseline_training.txt', 'w') as f:
#     f.write('id_gen_labels: ' + str(id_tr_labels) + '\n')
#     f.write('id_gen_pred_labels: ' + str(id_tr_pred_labels) + '\n')

# with open('./Label_compare/baseline_test.txt', 'w') as f:
#     f.write('id_gen_labels: ' + str(id_te_labels) + '\n')
#     f.write('id_gen_pred_labels: ' + str(id_te_pred_labels) + '\n')

