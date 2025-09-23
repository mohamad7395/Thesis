import sys
import os
import torch
import pandas as pd

torch.cuda.empty_cache()

root_path = '/home/s6moakba/InstructABSA'
sys.path.append(root_path)
os.chdir(root_path)

use_mps = True if torch.has_mps else False

from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator
from instructions import InstructionsHandler
from InstructABSA.extra_utils.set_up import train_val_split, get_train_args
from InstructABSA.extra_utils.preprocessing import format_atsc  

print("_________model_training_________")

epoch_adress = '/home/s6moakba/InstructABSA/epoch_metrics/text.csv'
model_out_path = './Models_baseline_ATSC_16'
model_checkpoint = 'allenai/tk-instruct-base-def-pos'

train_df = pd.read_csv("./Dataset/Benchmarks/SemEval16/Train/Restaurants_Train.csv")
test_df = pd.read_csv("./Dataset/Benchmarks/SemEval16/Test/Restaurants_Test.csv")

train_df, val_df = train_val_split(train_df)


# --- Format + filter inside format_atsc ---
instruct_handler = InstructionsHandler()
instruct_handler.load_instruction_set1()
loader = DatasetLoader(train_df, test_df, val_df)
loader = format_atsc(loader, instruct_handler)


# --- Train ---
t5_exp = T5Generator(model_checkpoint)
id_ds, id_tokenized_ds, _, _ = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs)

training_args = get_train_args(model_out_path, use_mps)
trainer = t5_exp.train_with_metrics(id_ds, id_tokenized_ds, epoch_adress,**training_args)
