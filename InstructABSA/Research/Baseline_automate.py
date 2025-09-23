import sys
import os
import torch
import pandas as pd

torch.cuda.empty_cache()

root_path = "/home/s6moakba/InstructABSA"
sys.path.append(root_path)
os.chdir(root_path)

use_mps = True if torch.has_mps else False

from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator
from instructions import InstructionsHandler
from InstructABSA.extra_utils.preprocessing import  (preprocess_data , parse_epoch_address_baseline)
from InstructABSA.extra_utils.set_up import train_val_split, get_train_args


MODEL_CHECKPOINTS = {
    "t5": "google-t5/t5-base",
    "tk": "allenai/tk-instruct-base-def-pos",
}
MODEL_OUT_PATH = "./MODEL_AAAAAAAA"


# ---------------------------
# Dataset mappings
# ---------------------------
TEST_MAPPING = {
    "16": "./Dataset/Benchmarks/SemEval16/Test/Restaurants_Test.csv",
    "15": "./Dataset/Benchmarks/SemEval15/Test/Restaurants_Test.csv",
    "14_r": "./Dataset/Benchmarks/SemEval14/Test/Restaurants_Test.csv",
    "14_l": "./Dataset/Benchmarks/SemEval14/Test/Laptops_Test.csv",
}

TRAIN_MAPPING = {
    "16": "./Dataset/Benchmarks/SemEval16/Train/Restaurants_Train.csv",
    "15": "./Dataset/Benchmarks/SemEval15/Train/Restaurants_Train.csv",
    "14_r": "./Dataset/Benchmarks/SemEval14/Train/Restaurants_Train.csv",
    "14_l": "./Dataset/Benchmarks/SemEval14/Train/Laptops_Train.csv",
}


# ---------------------------
# Main training loop
# ---------------------------
def automate_training(epoch_addresses):
    for addr in epoch_addresses:
        print("=" * 120)
        print(f"Processing address: {addr}")

        params = parse_epoch_address_baseline(addr)
        model_family, task, key = params["model_family"], params["task"], params["train_key"]

        CHECKPOINT = MODEL_CHECKPOINTS[model_family]
        # --- Load data ---
        train_df = pd.read_csv(TRAIN_MAPPING[key])
        test_df = pd.read_csv(TEST_MAPPING[key])
        train_df, val_df = train_val_split(train_df)

        print(f"Task: {task}, Dataset: {key}, Train size: {train_df.shape[0]}")

        # --- Preprocess ---
        loader = preprocess_data(model_family, task, train_df, test_df, val_df)

        # --- Train ---
        t5_exp = T5Generator(CHECKPOINT)
        id_ds, id_tokenized_ds, _, _ = loader.set_data_for_training_semeval(
            t5_exp.tokenize_function_inputs
        )

        training_args = get_train_args(MODEL_OUT_PATH, use_mps)

        trainer = t5_exp.train_with_metrics(id_ds, id_tokenized_ds, addr, **training_args)


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    epoch_addresses = [
        f"{root_path}/epoch_metrics/tk/16/ATSC/ATSC_t5_16.csv",
    ]
    automate_training(epoch_addresses)
