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
from InstructABSA.extra_utils.set_up import get_train_args
from InstructABSA.extra_utils.preprocessing import( preprocess_data , parse_epoch_address_lrs)

print("_________model_training_________")

MODEL_CHECKPOINTS = {
    "t5": "google-t5/t5-base",
    "tk": "allenai/tk-instruct-base-def-pos",
}

MODEL_OUT_PATH = "./MODEL_AAAAAAAA"

# ---------------------------
# Dataset mappings
# ---------------------------

TEST_MAPPING = {
    "16": f"{root_path}/Dataset/Benchmarks/SemEval16/Test/Restaurants_Test.csv",
    "15": f"{root_path}/Dataset/Benchmarks/SemEval15/Test/Restaurants_Test.csv",
    "14_r": f"{root_path}/Dataset/Benchmarks/SemEval14/Test/Restaurants_Test.csv",
    "14_l": f"{root_path}/Dataset/Benchmarks/SemEval14/Test/Laptops_Test.csv",
}

TR_MAPPING = {
    "agent_16": f"{root_path}/Dataset/Generated/LRS/16/agentic_lrs.csv",
    "prompting_16": f"{root_path}/Dataset/Generated/LRS/16/prompting_lrs.csv",
    "training_set_16": f"{root_path}/Dataset/Generated/LRS/16/training_set_lrs.csv",
    "agent_15": f"{root_path}/Dataset/Generated/LRS/15/agentic_lrs.csv",
    "prompting_15": f"{root_path}/Dataset/Generated/LRS/15/prompting_lrs.csv",
    "training_set_15": f"{root_path}/Dataset/Generated/LRS/15/training_set_lrs.csv",
    "agent_14_r": f"{root_path}/Dataset/Generated/LRS/14_r/agentic_lrs.csv",
    "prompting_14_r": f"{root_path}/Dataset/Generated/LRS/14_r/prompting_lrs.csv",
    "training_set_14_r": f"{root_path}/Dataset/Generated/LRS/14_r/training_set_lrs.csv",
    "agent_14_l": f"{root_path}/Dataset/Generated/LRS/14_l/agentic_lrs.csv",
    "prompting_14_l": f"{root_path}/Dataset/Generated/LRS/14_l/prompting_lrs.csv",
    "training_set_14_l": f"{root_path}/Dataset/Generated/LRS/14_l/training_set_lrs.csv",
}





# ---------------------------
# Training loop
# ---------------------------
def automate_training(epoch_addresses):
    for addr in epoch_addresses:
        print("=" * 120)
        print(f"Processing: {addr}")

        params = parse_epoch_address_lrs(addr)
        model_family = params["model_family"]
        task = params["task"]
        train_key = params["train_key"]
        test_key = params["test_key"]

        print(f"Model: {model_family}, Task: {task}, Train Key: {train_key}, Test Key: {test_key}")

        train_df = pd.read_csv(TR_MAPPING[train_key])
        test_df = pd.read_csv(TEST_MAPPING[test_key])
        val_df = train_df.copy()  

        print("Train size:", train_df.shape, "Test size:", test_df.shape)

        loader = preprocess_data(model_family, task, train_df, test_df, val_df)

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
        f"{root_path}/epoch_metrics/LRS_final/tk/14_l/ATE/agent.csv",
        f"{root_path}/epoch_metrics/LRS_final/tk/14_l/ATE/prompting.csv",
        f"{root_path}/epoch_metrics/LRS_final/tk/14_l/ATSC/agent.csv",
        f"{root_path}/epoch_metrics/LRS_final/tk/14_l/ATSC/prompting.csv",
        f"{root_path}/epoch_metrics/LRS_final/tk/14_l/ASPE/agent.csv",
        f"{root_path}/epoch_metrics/LRS_final/tk/14_l/ASPE/prompting.csv",

    ]
    automate_training(epoch_addresses)
