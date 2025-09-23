import sys
import os
import re
import torch
import pandas as pd

torch.cuda.empty_cache()

# ---------------------------
# Project setup
# ---------------------------
root_path = "/home/s6moakba/InstructABSA"
sys.path.append(root_path)
os.chdir(root_path)

use_mps = True if torch.has_mps else False

from InstructABSA.data_prep import DatasetLoader
from InstructABSA.utils import T5Generator
from instructions import InstructionsHandler
from InstructABSA.extra_utils.set_up import train_val_split, get_train_args
from InstructABSA.extra_utils.preprocessing import ( preprocess_data , parse_epoch_address_experiment , load_augmented_data)


print("_________model_training_________")

MODEL_CHECKPOINTS = {
    "t5": "google-t5/t5-base",
    "tk": "allenai/tk-instruct-base-def-pos",
}

MODEL_OUT_PATH = "./Exp_14_r"


TRAIN_FILE = "./Dataset/Benchmarks/SemEval14/Train/Restaurants_Train.csv"
TEST_FILE = "./Dataset/Benchmarks/SemEval14/Test/Restaurants_Test.csv"
TR_SIZE = 2700

AUG_MAPPING = {
    "agent": f"{root_path}/Dataset/Generated/AUG/14_r/Agentic_14_r_5k.csv",
    "prompting": f"{root_path}/Dataset/Generated/AUG/14_r/Agentic_14_r_5k.csv",
}


def load_augmented_data(strategy, ratio):
    """Return augmented dataframe based on strategy + ratio."""
    gen_file = AUG_MAPPING[strategy]
    aug_df = pd.read_csv(gen_file)

    if ratio == 1:
        return aug_df.sample(n=TR_SIZE, random_state=41)
    elif ratio == 2:
        return aug_df
    else:
        raise ValueError(f"Unsupported ratio: {ratio}")



# ---------------------------
# Training loop
# ---------------------------
def automate_training(epoch_addresses):
    for addr in epoch_addresses:
        print("=" * 120)
        print(f"Processing: {addr}")

        params = parse_epoch_address_experiment(addr)
        model_family = params["model_family"]
        task = params["task"]
        strategy = params["strategy"]
        train_include = params["train_include"]
        ratio = params["ratio"]

        # Load base train/test
        train_df = pd.read_csv(TRAIN_FILE)
        test_df = pd.read_csv(TEST_FILE)
        train_df, val_df = train_val_split(train_df)  # default 0.1 split

        # Load augmented
        aug_df = load_augmented_data(strategy, ratio)

        if train_include:
            train_df = pd.concat([train_df, aug_df], ignore_index=True)
        else:
            train_df = aug_df

        print(
            f"Model: {model_family}, Task: {task}, Strategy: {strategy}, Train Include: {train_include}, Ratio: {ratio}"
        )

        # Preprocess
        loader = preprocess_data(model_family, task, train_df, test_df, val_df)

        # Train
        checkpoint = MODEL_CHECKPOINTS[model_family]
        t5_exp = T5Generator(checkpoint)
        id_ds, id_tokenized_ds, _, _ = loader.set_data_for_training_semeval(
            t5_exp.tokenize_function_inputs
        )

        training_args = get_train_args(MODEL_OUT_PATH, use_mps)
        t5_exp.train_with_metrics(id_ds, id_tokenized_ds, addr, **training_args)


# ---------------------------
# Entrypoint
# ---------------------------
if __name__ == "__main__":
    epoch_addresses = [
        f"{root_path}/epoch_metrics/t5/14_r/ATE/ATE_agent_with_train_x1.csv",
        f"{root_path}/epoch_metrics/t5/14_r/ATE/ATE_agent_with_train_x2.csv",
        f"{root_path}/epoch_metrics/t5/14_r/ATE/ATE_agent_no_train_x1.csv",
        f"{root_path}/epoch_metrics/t5/14_r/ATE/ATE_agent_no_train_x2.csv",
        f"{root_path}/epoch_metrics/t5/14_r/ATSC/ATSC_agent_with_train_x1.csv",
        f"{root_path}/epoch_metrics/t5/14_r/ATSC/ATSC_agent_with_train_x2.csv",
        f"{root_path}/epoch_metrics/t5/14_r/ATSC/ATSC_agent_no_train_x1.csv",
        f"{root_path}/epoch_metrics/t5/14_r/ATSC/ATSC_agent_no_train_x2.csv",
        f"{root_path}/epoch_metrics/t5/14_r/ASPE/ASPE_agent_with_train_x1.csv",
        f"{root_path}/epoch_metrics/t5/14_r/ASPE/ASPE_agent_with_train_x2.csv",
        f"{root_path}/epoch_metrics/t5/14_r/ASPE/ASPE_agent_no_train_x1.csv",
        f"{root_path}/epoch_metrics/t5/14_r/ASPE/ASPE_agent_no_train_x2.csv",
    ]
    automate_training(epoch_addresses)
