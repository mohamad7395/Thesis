
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
import re
import pandas as pd



print('_________model_training_________')

model_checkpoint = 'google-t5/t5-base'
model_out_path = './MODEL_AAAAAAAA'


test_mapping= {
    '16': '/home/s6moakba/InstructABSA/Dataset/SemEval16/Test/Restaurants_Test.csv',
    '15': '/home/s6moakba/InstructABSA/Dataset/SemEval15/Test/Restaurants_Test.csv',
    '14_r': '/home/s6moakba/InstructABSA/Dataset/SemEval14/Test/Restaurants_Test.csv',
    '14_l': '/home/s6moakba/InstructABSA/Dataset/SemEval14/Test/Laptops_Test.csv',
}

tr_mapping = {
    '16': '/home/s6moakba/InstructABSA/Dataset/SemEval16/Train/Restaurants_Train.csv',
    '15': '/home/s6moakba/InstructABSA/Dataset/SemEval15/Train/Restaurants_Train.csv',
    '14_r': '/home/s6moakba/InstructABSA/Dataset/SemEval14/Train/Restaurants_Train.csv',
    '14_l': '/home/s6moakba/InstructABSA/Dataset/SemEval14/Train/Laptops_Train.csv',

}
VALIDATION_RATIO = 0.1


def get_training_args():
    training_args = {
        'output_dir': model_out_path,
        'logging_steps': 100,
        'report_to': 'tensorboard',
        'evaluation_strategy': 'epoch',
        'learning_rate': 5e-5,
        'lr_scheduler_type': 'cosine',
        'per_device_train_batch_size': 8,
        'per_device_eval_batch_size': 16,
        'num_train_epochs': 20,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,

        'save_strategy': 'no',
        'save_steps': 0,
        'save_total_limit': 0,

        'load_best_model_at_end': False,
        'push_to_hub': False,
        'eval_accumulation_steps': 1,
        'predict_with_generate': True,
        'use_mps_device': use_mps
    }
    return training_args

def parse_epoch_address(path):

    # 1) Strip off the filename and its extension:
    filename = os.path.basename(path)         # e.g. "training_set.csv"
    base, _ext = os.path.splitext(filename)   # e.g. ("training_set", ".csv")

    # 2) The folder one level up from the file:
    task = os.path.basename(os.path.dirname(path)) 

    # 3) The folder two levels up:
    dataset = os.path.basename(
        os.path.dirname(os.path.dirname(path))
    )

    # dataset_key = base + '_' + dataset

    return {
        "task": task,
        "train_key": dataset,
        'test_key': dataset
    }



def automate_training(epoch_addresses):

    for addr in epoch_addresses:
        print('_'*200)
        print(f"Processing address: {addr}")

        params = parse_epoch_address(addr)

        task = params['task']
        train_key = params['train_key']
        test_key = params['test_key']

        id_train_file_path = tr_mapping[train_key]
        id_test_file_path = test_mapping[test_key]

        id_tr_df = pd.read_csv(id_train_file_path)
        id_te_df = pd.read_csv(id_test_file_path)


        num_samples = len(id_tr_df)
        num_validation_samples = int(VALIDATION_RATIO * num_samples)

        id_val_df = id_tr_df.sample(n=num_validation_samples, random_state=42)
        id_tr_df = id_tr_df.drop(id_val_df.index)   

        params = parse_epoch_address(addr)

        task = params['task']
        train_key = params['train_key']
        test_key = params['test_key']

        print(f"Task: {task}, Train Key: {train_key}, Test Key: {test_key}")

        instruct_handler = InstructionsHandler()
        instruct_handler.load_instruction_set1()

        if task == "ATE":
            loader = DatasetLoader(train_df_id = id_tr_df, test_df_id = id_te_df, val_df_id =id_val_df )
            if loader.train_df_id is not None:
                loader.train_df_id = loader.create_data_in_ate_format(loader.train_df_id, 'term', 'raw_text', 'aspectTerms', '', '')
            if loader.test_df_id is not None:
                loader.test_df_id = loader.create_data_in_ate_format(loader.test_df_id, 'term', 'raw_text', 'aspectTerms','','')
            if loader.val_df_id is not None:
                loader.val_df_id = loader.create_data_in_ate_format(loader.val_df_id, 'term', 'raw_text', 'aspectTerms', '', '')

        elif task == "ATSC":
            loader = DatasetLoader(train_df_id = id_tr_df, test_df_id = id_te_df, val_df_id =id_val_df )
            if loader.train_df_id is not None:
                loader.train_df_id = loader.create_data_in_atsc_format(loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', '','','')
            if loader.test_df_id is not None:
                loader.test_df_id = loader.create_data_in_atsc_format(loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', '','','')
            if loader.val_df_id is not None:
                loader.val_df_id = loader.create_data_in_atsc_format(loader.val_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', '','','')
                
            loader.train_df_id  = loader.train_df_id[loader.train_df_id['labels'] != 'none']
            loader.test_df_id  = loader.test_df_id[loader.test_df_id['labels'] != 'none']
            loader.val_df_id  = loader.val_df_id[loader.val_df_id['labels'] != 'none']

            loader.train_df_id  = loader.train_df_id[loader.train_df_id['labels'] != 'conflict']
            loader.test_df_id  = loader.test_df_id[loader.test_df_id['labels'] != 'conflict']
            loader.val_df_id  = loader.val_df_id[loader.val_df_id['labels'] != 'conflict']
        
        elif task == "ASPE":
            loader = DatasetLoader(train_df_id = id_tr_df, test_df_id = id_te_df, val_df_id =id_val_df )
            if loader.train_df_id is not None:
                loader.train_df_id = loader.create_data_in_aspe_format(loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', '', '')
            if loader.test_df_id is not None:
                loader.test_df_id = loader.create_data_in_aspe_format(loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', '', '')
            if loader.val_df_id is not None:
                loader.val_df_id = loader.create_data_in_aspe_format(loader.val_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', '', '')

        
        t5_exp = T5Generator(model_checkpoint)
        id_ds, id_tokenized_ds, _, _ = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs)
        training_args = get_training_args()
        model_trainer = t5_exp.train_with_metrics(id_ds, id_tokenized_ds, addr,**training_args)



if __name__ == "__main__":
    epoch_addresses = [       
        "/home/s6moakba/InstructABSA/epoch_metrics/t5/16/ATSC/ATSC_t5_16.csv",
        "/home/s6moakba/InstructABSA/epoch_metrics/t5/16/ASPE/ASPE_t5_16.csv",

        "/home/s6moakba/InstructABSA/epoch_metrics/t5/15/ATE/ATE_t5_15.csv",
        "/home/s6moakba/InstructABSA/epoch_metrics/t5/15/ATSC/ATSC_t5_15.csv",
        "/home/s6moakba/InstructABSA/epoch_metrics/t5/15/ASPE/ASPE_t5_15.csv",

        "/home/s6moakba/InstructABSA/epoch_metrics/t5/14_r/ATE/ATE_t5_14_r.csv",
        "/home/s6moakba/InstructABSA/epoch_metrics/t5/14_r/ATSC/ATSC_t5_14_r.csv",
        "/home/s6moakba/InstructABSA/epoch_metrics/t5/14_r/ASPE/ASPE_t5_14_r.csv",

        "/home/s6moakba/InstructABSA/epoch_metrics/t5/14_l/ATE/ATE_t5_14_l.csv",
        "/home/s6moakba/InstructABSA/epoch_metrics/t5/14_l/ATSC/ATSC_t5_14_l.csv",
        "/home/s6moakba/InstructABSA/epoch_metrics/t5/14_l/ASPE/ASPE_t5_14_l.csv",
    ]
    
    automate_training(epoch_addresses)