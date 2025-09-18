
print('_______importing_______')
import sys
import os

import torch

torch.cuda.empty_cache()


# Add the root path to sys.path
root_path = '/home/s6moakba/InstructABSA'
sys.path.append(root_path)
os.chdir(root_path)
    
os.chdir(root_path)
use_mps = True if torch.has_mps else False

from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator, T5Classifier
from instructions import InstructionsHandler

import pandas as pd



print('_________model_training_________')
model_checkpoint = 'google-t5/t5-base'
model_out_path = './MODEL_AAAAAAAA'


id_train_file_path = './Dataset/SemEval16/Train/Restaurants_Train.csv'
id_test_file_path = './Dataset/SemEval16/Test/Restaurants_Test.csv'

id_tr_df = pd.read_csv(id_train_file_path)




validation_ratio = 0.1
num_samples = len(id_tr_df)
num_validation_samples = int(validation_ratio * num_samples)

id_val_df = id_tr_df.sample(n=num_validation_samples, random_state=42)
id_tr_df = id_tr_df.drop(id_val_df.index)




print('size of training set : ',id_tr_df.shape)


id_te_df = pd.read_csv(id_test_file_path)

print("_"*50)



# Get the input text into the required format using Instructions
instruct_handler = InstructionsHandler()

instruct_handler.load_instruction_set1()

# Set instruction_set1 for InstructABSA-1 and instruction_set2 for InstructABSA-2
loader = DatasetLoader(train_df_id = id_tr_df, test_df_id = id_te_df, val_df_id =id_val_df )
# Set bos_instruct1 for lapt14 and bos_instruct2 for rest14. For other datasets, modify the insructions.py file.
if loader.train_df_id is not None:
    loader.train_df_id = loader.create_data_in_atsc_format(loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect','','','')
if loader.test_df_id is not None:
    loader.test_df_id = loader.create_data_in_atsc_format(loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', '','','')
if loader.val_df_id is not None:
    loader.val_df_id = loader.create_data_in_atsc_format(loader.val_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect','','','')

    
# Data preprocess
loader.train_df_id  = loader.train_df_id[loader.train_df_id['labels'] != 'none']
loader.test_df_id  = loader.test_df_id[loader.test_df_id['labels'] != 'none']
loader.val_df_id  = loader.val_df_id[loader.val_df_id['labels'] != 'none']

loader.train_df_id  = loader.train_df_id[loader.train_df_id['labels'] != 'conflict']
loader.test_df_id  = loader.test_df_id[loader.test_df_id['labels'] != 'conflict']
loader.val_df_id  = loader.val_df_id[loader.val_df_id['labels'] != 'conflict']

print(loader.train_df_id.shape)


# Create T5 utils object
t5_exp = T5Generator(model_checkpoint)

# Tokenize Dataset
id_ds, id_tokenized_ds, ood_ds, ood_tokenized_ds = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs)

# Training arguments
training_args = {
    'output_dir':model_out_path,
    'logging_steps':100,
    'report_to' :'tensorboard',
    'evaluation_strategy':'epoch',
    'learning_rate':5e-4,
    'lr_scheduler_type':'cosine',
    'per_device_train_batch_size':8,
    'per_device_eval_batch_size':16,
    'num_train_epochs':20,
    'weight_decay':0.01,
    'warmup_ratio':0.1,
    'save_strategy':'no',
    'load_best_model_at_end':False,
    'push_to_hub':False,
    'eval_accumulation_steps':1,
    'predict_with_generate':True,
    'use_mps_device':use_mps
    }    


model_trainer = t5_exp.train_with_metrics(id_ds, id_tokenized_ds, epoch_adress,**training_args)
