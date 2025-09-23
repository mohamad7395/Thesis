import pandas as pd

def train_val_split(df: pd.DataFrame, validation_ratio: float = 0.1, seed: int = 42):
    """
    Split a DataFrame into train and validation sets.

    Args:
        df (pd.DataFrame): Input dataframe.
        validation_ratio (float): Fraction of validation data.
        seed (int): Random seed for reproducibility.

    Returns:
        (train_df, val_df): Two pandas DataFrames.
    """
    num_samples = len(df)
    num_val_samples = int(validation_ratio * num_samples)
    val_df = df.sample(n=num_val_samples, random_state=seed)
    train_df = df.drop(val_df.index)
    return train_df, val_df

def get_train_args(model_out_path , use_mps):
    training_args = {
        'output_dir':model_out_path,
        'logging_steps':100,
        'report_to' :'tensorboard',
        'evaluation_strategy':'epoch',
        'learning_rate':5e-5,
        'lr_scheduler_type':'cosine',
        'per_device_train_batch_size':8,
        'per_device_eval_batch_size':16,
        'num_train_epochs':20,
        'weight_decay':0.01,
        'warmup_ratio':0.1,
        'save_strategy':'no',
        'save_steps':0,
        'save_total_limit':0,
        'load_best_model_at_end':False,
        'push_to_hub':False,
        'eval_accumulation_steps':1,
        'predict_with_generate':True,
        'use_mps_device':use_mps
        }   
    return training_args