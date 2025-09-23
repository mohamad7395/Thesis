from InstructABSA.data_prep import DatasetLoader
from instructions import InstructionsHandler
import os, re
import pandas as pd

def format_ate(loader, instruct_handler):
    """
    Apply ATE formatting to train/val/test dataframes in a DatasetLoader.
    """
    if loader.train_df_id is not None:
        loader.train_df_id = loader.create_data_in_ate_format(
            loader.train_df_id, 'term', 'raw_text', 'aspectTerms',
            instruct_handler.ate['bos_instruct2'], instruct_handler.ate['eos_instruct']
        )
    if loader.test_df_id is not None:
        loader.test_df_id = loader.create_data_in_ate_format(
            loader.test_df_id, 'term', 'raw_text', 'aspectTerms',
            instruct_handler.ate['bos_instruct2'], instruct_handler.ate['eos_instruct']
        )
    if loader.val_df_id is not None:
        loader.val_df_id = loader.create_data_in_ate_format(
            loader.val_df_id, 'term', 'raw_text', 'aspectTerms',
            instruct_handler.ate['bos_instruct2'], instruct_handler.ate['eos_instruct']
        )
    return loader


def format_atsc(loader, instruct_handler):
    """
    Apply ATSC formatting to train/val/test dataframes in a DatasetLoader.
    """
    if loader.train_df_id is not None:
        loader.train_df_id = loader.create_data_in_atsc_format(
            loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect',
            instruct_handler.atsc['bos_instruct2'], instruct_handler.atsc['delim_instruct'],
            instruct_handler.atsc['eos_instruct']
        )
    if loader.test_df_id is not None:
        loader.test_df_id = loader.create_data_in_atsc_format(
            loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect',
            instruct_handler.atsc['bos_instruct2'], instruct_handler.atsc['delim_instruct'],
            instruct_handler.atsc['eos_instruct']
        )
    if loader.val_df_id is not None:
        loader.val_df_id = loader.create_data_in_atsc_format(
            loader.val_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect',
            instruct_handler.atsc['bos_instruct2'], instruct_handler.atsc['delim_instruct'],
            instruct_handler.atsc['eos_instruct']
        )

    for split_name in ["train_df_id", "test_df_id", "val_df_id"]:
        df = getattr(loader, split_name)
        if df is not None:
            df = df[df["labels"] != "none"]
            df = df[df["labels"] != "conflict"]
            setattr(loader, split_name, df)
    return loader

def format_aspe(loader, instruct_handler):
    """
    Apply ASPE formatting to train/val/test dataframes in a DatasetLoader.
    """
    if loader.train_df_id is not None:
        loader.train_df_id = loader.create_data_in_aspe_format(
            loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms',
            instruct_handler.aspe['bos_instruct2'], instruct_handler.aspe['eos_instruct']
        )
    if loader.test_df_id is not None:
        loader.test_df_id = loader.create_data_in_aspe_format(
            loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms',
            instruct_handler.aspe['bos_instruct2'], instruct_handler.aspe['eos_instruct']
        )
    if loader.val_df_id is not None:
        loader.val_df_id = loader.create_data_in_aspe_format(
            loader.val_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms',
            instruct_handler.aspe['bos_instruct2'], instruct_handler.aspe['eos_instruct']
        )
    return loader


def format_ate_t5(loader):
    """
    Apply T5 baseline formatting (no BOS/EOS instructions).
    """
    if loader.train_df_id is not None:
        loader.train_df_id = loader.create_data_in_ate_format(
            loader.train_df_id, 'term', 'raw_text', 'aspectTerms', '', ''
        )
    if loader.test_df_id is not None:
        loader.test_df_id = loader.create_data_in_ate_format(
            loader.test_df_id, 'term', 'raw_text', 'aspectTerms', '', ''
        )
    if loader.val_df_id is not None:
        loader.val_df_id = loader.create_data_in_ate_format(
            loader.val_df_id, 'term', 'raw_text', 'aspectTerms', '', ''
        )
    return loader

def format_atsc_t5(loader):
    """
    Apply ATSC formatting for T5 baseline (no BOS/DELIM/EOS instructions)
    + filter out 'none' and 'conflict' labels.
    """
    if loader.train_df_id is not None:
        loader.train_df_id = loader.create_data_in_atsc_format(
            loader.train_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', '', '', ''
        )
    if loader.test_df_id is not None:
        loader.test_df_id = loader.create_data_in_atsc_format(
            loader.test_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', '', '', ''
        )
    if loader.val_df_id is not None:
        loader.val_df_id = loader.create_data_in_atsc_format(
            loader.val_df_id, 'aspectTerms', 'term', 'raw_text', 'aspect', '', '', ''
        )

    # 🔹 Apply label filtering
    for split_name in ["train_df_id", "test_df_id", "val_df_id"]:
        df = getattr(loader, split_name)
        if df is not None:
            df = df[df["labels"] != "none"]
            df = df[df["labels"] != "conflict"]
            setattr(loader, split_name, df)

    return loader

def format_aspe_t5(loader):
    """
    Apply ASPE formatting for T5 baseline (no BOS/EOS instructions).
    """
    if loader.train_df_id is not None:
        loader.train_df_id = loader.create_data_in_aspe_format(
            loader.train_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', '', ''
        )
    if loader.test_df_id is not None:
        loader.test_df_id = loader.create_data_in_aspe_format(
            loader.test_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', '', ''
        )
    if loader.val_df_id is not None:
        loader.val_df_id = loader.create_data_in_aspe_format(
            loader.val_df_id, 'term', 'polarity', 'raw_text', 'aspectTerms', '', ''
        )

    return loader

def preprocess_data(model_family, task, train_df, test_df, val_df):
    """Route to correct formatting function."""
    instruct_handler = InstructionsHandler()
    instruct_handler.load_instruction_set1()
    loader = DatasetLoader(train_df, test_df, val_df)

    if model_family == "t5":
        if task == "ATE":
            return format_ate_t5(loader)
        elif task == "ATSC":
            return format_atsc_t5(loader)
        elif task == "ASPE":
            return format_aspe_t5(loader)
    elif model_family == "tk":
        if task == "ATE":
            return format_ate(loader, instruct_handler)
        elif task == "ATSC":
            return format_atsc(loader, instruct_handler)
        elif task == "ASPE":
            return format_aspe(loader, instruct_handler)

    raise ValueError(f"Unsupported combo: {model_family}, {task}")


def parse_epoch_address_baseline(path):

    task = os.path.basename(os.path.dirname(path))               # ATSC, ATE, ASPE
    dataset = os.path.basename(os.path.dirname(os.path.dirname(path)))  # 14_l, 15, 16
    model_family = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))  # t5, tk, etc.

    return {
        "model_family": model_family,
        "task": task,
        "train_key": dataset,
        "test_key": dataset,
    }

def parse_epoch_address_experiment(epoch_address: str) -> dict:
    """
    Parse filenames like:
      /.../epoch_metrics/t5/14_r/ASPE/ASPE_agent_with_train_x2.csv
    -> {
         "model_family": "t5",
         "task": "ASPE",
         "strategy": "agent",
         "train_include": True,
         "ratio": 2
       }
    """
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

    model_family = os.path.basename(
        os.path.dirname(os.path.dirname(os.path.dirname(epoch_address)))
    )  # e.g. "t5" or "tk"

    return {
        "model_family": model_family,
        "task": task,
        "strategy": strategy,
        "train_include": train_include,
        "ratio": ratio,
    }


def parse_epoch_address_lrs(path):
    """
    Example:
      /.../epoch_metrics/LRS_final/t5/14_l/ATE/agent.csv
    -> {
         "model_family": "t5",
         "task": "ATE",
         "train_key": "agent_14_l",
         "test_key": "14_l"
       }
    """
    fname = os.path.splitext(os.path.basename(path))[0]  # "agent"
    task = os.path.basename(os.path.dirname(path))       # "ATE"
    dataset = os.path.basename(os.path.dirname(os.path.dirname(path)))  # "14_l"
    model_family = os.path.basename(
        os.path.dirname(os.path.dirname(os.path.dirname(path)))
    )  # e.g. "t5" or "tk"

    train_key = f"{fname}_{dataset}"  
    return {
        "model_family": model_family,
        "task": task,
        "train_key": train_key,
        "test_key": dataset,
    }


