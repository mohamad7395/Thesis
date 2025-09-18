
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


id_train_file_path = './Dataset/SemEval14/Train/Restaurants_Train.csv'
id_test_file_path = './Dataset/SemEval14/Test/Restaurants_Test.csv'
VALIDATION_RATIO = 0.1

TR_SIZE = 2700

id_tr_df = pd.read_csv(id_train_file_path)
id_te_df = pd.read_csv(id_test_file_path)

num_samples = len(id_tr_df)
num_validation_samples = int(VALIDATION_RATIO * num_samples)

id_val_df = id_tr_df.sample(n=num_validation_samples, random_state=42)
id_tr_df = id_tr_df.drop(id_val_df.index)



aug_mapping = {
    'agent': "/home/s6moakba/InstructABSA/Dataset/AUG/14_r/Agentic_14_r_5k.csv",
    'prompting': '/home/s6moakba/InstructABSA/Dataset/AUG/14_r/Agentic_14_r_5k.csv',
}

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

def parse_epoch_address(epoch_address: str) -> dict:

    fname = os.path.splitext(os.path.basename(epoch_address))[0]
    
    parts = fname.split("_")
    
    task = parts[0]
    strategy = parts[1]
    
 
    train_include = "with" in parts
    
    ratio = None
    for p in parts:
        m = re.fullmatch(r"x(\d+)", p)
        if m:
            ratio = int(m.group(1))
            break
    
    return {
        "task": task,
        "strategy": strategy,
        "train_include": train_include,
        "ratio": ratio,
    }



def automate_training(epoch_addresses):

    for addr in epoch_addresses:
        print('_'*200)
        print(f"Processing address: {addr}")
        id_tr_df = pd.read_csv(id_train_file_path)
        id_te_df = pd.read_csv(id_test_file_path)

        num_samples = len(id_tr_df)
        num_validation_samples = int(VALIDATION_RATIO * num_samples)

        id_val_df = id_tr_df.sample(n=num_validation_samples, random_state=42)
        id_tr_df = id_tr_df.drop(id_val_df.index)   

        params = parse_epoch_address(addr)

        task = params["task"]
        aug_strategy = params["strategy"]
        train_include = params["train_include"]
        ratio = params["ratio"]

        print(f"Task: {task}, Strategy: {aug_strategy}, Train Include: {train_include}, Ratio: {ratio}")


        id_train_generated_file_path = aug_mapping[aug_strategy]
        id_tr_df_gen = pd.read_csv(id_train_generated_file_path)

        if ratio == 1:
            id_tr_df_gen_sampled = id_tr_df_gen.sample(n=TR_SIZE, random_state=41)
        elif ratio == 2:
            id_tr_df_gen_sampled = id_tr_df_gen
        

        if train_include:
            id_tr_df = pd.concat([id_tr_df, id_tr_df_gen_sampled], ignore_index=True, axis=0)
        elif not train_include:
            id_tr_df = id_tr_df_gen_sampled

        print('size of training set : ',id_tr_df.shape)

        instruct_handler = InstructionsHandler()
        instruct_handler.load_instruction_set1()

        if task == "ATE":
            loader = DatasetLoader(train_df_id = id_tr_df, test_df_id = id_te_df, val_df_id =id_val_df )
            if loader.train_df_id is not None:
                loader.train_df_id = loader.create_data_in_ate_format(loader.train_df_id, 'term', 'raw_text', 'aspectTerms', '','')
            if loader.test_df_id is not None:
                loader.test_df_id = loader.create_data_in_ate_format(loader.test_df_id, 'term', 'raw_text', 'aspectTerms', '','')
            if loader.val_df_id is not None:
                loader.val_df_id = loader.create_data_in_ate_format(loader.val_df_id, 'term', 'raw_text', 'aspectTerms', '','')

        elif task == "ATSC":
            loader = DatasetLoader(train_df_id = id_tr_df, test_df_id = id_te_df, val_df_id =id_val_df )
            if loader.train_df_id is not None:
                loader.train_df_id = loader.create_data_in_atsc_format(loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect',  '','', '')
            if loader.test_df_id is not None:
                loader.test_df_id = loader.create_data_in_atsc_format(loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect','','', '')
            if loader.val_df_id is not None:
                loader.val_df_id = loader.create_data_in_atsc_format(loader.val_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', '','', '')
                
            loader.train_df_id  = loader.train_df_id[loader.train_df_id['labels'] != 'none']
            loader.test_df_id  = loader.test_df_id[loader.test_df_id['labels'] != 'none']
            loader.val_df_id  = loader.val_df_id[loader.val_df_id['labels'] != 'none']

            loader.train_df_id  = loader.train_df_id[loader.train_df_id['labels'] != 'conflict']
            loader.test_df_id  = loader.test_df_id[loader.test_df_id['labels'] != 'conflict']
            loader.val_df_id  = loader.val_df_id[loader.val_df_id['labels'] != 'conflict']
        
        elif task == "ASPE":
            loader = DatasetLoader(train_df_id = id_tr_df, test_df_id = id_te_df, val_df_id =id_val_df )
            if loader.train_df_id is not None:
                loader.train_df_id = loader.create_data_in_aspe_format(loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms','', '')
            if loader.test_df_id is not None:
                loader.test_df_id = loader.create_data_in_aspe_format(loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', '', '')
            if loader.val_df_id is not None:
                loader.val_df_id = loader.create_data_in_aspe_format(loader.val_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms',  '', '')

        
        t5_exp = T5Generator(model_checkpoint)
        id_ds, id_tokenized_ds, _, _ = loader.set_data_for_training_semeval(t5_exp.tokenize_function_inputs)
        training_args = get_training_args()
        model_trainer = t5_exp.train_with_metrics(id_ds, id_tokenized_ds, addr,**training_args)



if __name__ == "__main__":
    epoch_addresses = [       
       "/home/s6moakba/InstructABSA/epoch_metrics/t5/14_r/ATE/ATE_agent_with_train_x1.csv",
       "/home/s6moakba/InstructABSA/epoch_metrics/t5/14_r/ATSC/ATSC_agent_with_train_x1.csv",
       "/home/s6moakba/InstructABSA/epoch_metrics/t5/14_r/ASPE/ASPE_agent_with_train_x1.csv",

    ]
    
    automate_training(epoch_addresses)