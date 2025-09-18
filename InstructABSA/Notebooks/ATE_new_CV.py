
print('_______importing_______')
import sys
import os

import torch
import pandas as pd
from sklearn.model_selection import KFold

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
task_name = 'ate'
experiment_name = 'lapt2014_iabsa1'
model_checkpoint = 'allenai/tk-instruct-base-def-pos'
model_out_base_path = './Models_baseline_16'
num_folds = 10




id_train_file_path = './Dataset/SemEval16/Train/Restaurants_Train.csv'
id_test_file_path = './Dataset/SemEval16/Test/Restaurants_Test.csv'

id_tr_df_full = pd.read_csv(id_train_file_path)
id_te_df = pd.read_csv(id_test_file_path)
print("_"*50)

validation_ratio = 0.1
num_samples = len(id_tr_df_full)
num_validation_samples = int(validation_ratio * num_samples)

og_id_val_df = id_tr_df_full.sample(n=num_validation_samples, random_state=42)
id_tr_df_full = id_tr_df_full.drop(og_id_val_df.index)

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
fold_num = 1

for train_index, val_index in kf.split(id_tr_df_full):
    print(f"========== Fold {fold_num} / {num_folds} ==========")
    
    # ------------------ 1. Split the data ------------------
    id_tr_df   = id_tr_df_full.iloc[train_index].copy()
    id_val_df  = id_tr_df_full.iloc[val_index].copy()
    print(f"Train size: {id_tr_df.shape[0]} | Val size: {id_val_df.shape[0]}")

    instruct_handler = InstructionsHandler()

    # Set instruction_set1 for InstructABSA-1 and instruction_set2 for InstructABSA-2
    instruct_handler.load_instruction_set1()

    # Set bos_instruct1 for lapt14 and bos_instruct2 for rest14. For other datasets, modify the insructions.py file.
    loader = DatasetLoader(train_df_id = id_tr_df, test_df_id = id_te_df, val_df_id =id_val_df )
    if loader.train_df_id is not None:
        loader.train_df_id = loader.create_data_in_ate_format(loader.train_df_id, 'term', 'raw_text', 'aspectTerms', instruct_handler.ate['bos_instruct2'], instruct_handler.ate['eos_instruct'])
    if loader.test_df_id is not None:
        loader.test_df_id = loader.create_data_in_ate_format(loader.test_df_id, 'term', 'raw_text', 'aspectTerms', instruct_handler.ate['bos_instruct2'], instruct_handler.ate['eos_instruct'])
    if loader.val_df_id is not None:
        loader.val_df_id = loader.create_data_in_ate_format(loader.val_df_id, 'term', 'raw_text', 'aspectTerms', instruct_handler.ate['bos_instruct2'], instruct_handler.ate['eos_instruct'])


    # Create T5 utils object
    t5_exp = T5Generator(model_checkpoint)

    # Tokenize Dataset
    id_ds, id_tokenized_ds, ood_ds, ood_tokenized_ds = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs)

    fold_model_out_path = os.path.join(
        model_out_base_path, 
        task_name, 
        f"{model_checkpoint.replace('/', '')}-{experiment_name}-fold{fold_num}"
    )


    # Training arguments
    training_args = {
        'output_dir':fold_model_out_path,
        'logging_steps':100,
        'report_to' :'tensorboard',
        'evaluation_strategy':'epoch',
        'learning_rate':5e-5,
        'lr_scheduler_type':'cosine',
        'per_device_train_batch_size':8,
        'per_device_eval_batch_size':16,
        'num_train_epochs':4,
        'weight_decay':0.01,
        'warmup_ratio':0.1,
        'save_strategy':'no',
        'load_best_model_at_end':False,
        'push_to_hub':False,
        'eval_accumulation_steps':1,
        'predict_with_generate':True,
        'use_mps_device':use_mps
        }    


    model_trainer = t5_exp.train(id_tokenized_ds,**training_args)

    print('_________model_inference_________')

    # Model inference - Loading from Checkpoint
    t5_exp = T5Generator(fold_model_out_path)

    id_val_pred_labels = t5_exp.get_labels(tokenized_dataset = id_tokenized_ds, sample_set = 'validation',  batch_size = 16)
    id_val_labels = [i.strip() for i in id_ds['validation']['labels']]

    print('_________model_evaluation_________')
    p, r, f1, _ = t5_exp.get_metrics(id_val_labels, id_val_pred_labels)
    print(f"Precision: {p} | Recall: {r} | F1: {f1}")

    id_val_df['prediction'] = id_val_pred_labels

    val_output_path = os.path.join('/home/s6moakba/InstructABSA/CV_validation/res_16_ate_1_fold/', f"val_predictions_fold_{fold_num}.csv")
    id_val_df.to_csv(val_output_path, index=False)
    print(f"Saved validation predictions to {val_output_path}")
    
    # ------------------ 9. Move to next fold ------------------
    fold_num += 1
    print("==============================================")
    