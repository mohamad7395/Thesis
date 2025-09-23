import pandas as pd
import matplotlib.pyplot as plt

def generate_compare_f1(file_paths, sample_set, column_names):
    """
    Generates a DataFrame comparing F1 scores across multiple datasets.

    Parameters:
        file_paths (list): List of file paths to CSV files.
        sample_set (str): The name of the sample set to filter (e.g., 'test').
        column_names (list): List of column names for the resulting DataFrame.

    Returns:
        pd.DataFrame: A DataFrame with F1 scores for the specified sample set.
    """
    f1_scores = []
    
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        f1_scores.append(data[data["sample_set"] == sample_set]['f1'].reset_index(drop=True))
    
    compare_f1 = pd.concat(f1_scores, axis=1)
    compare_f1.columns = column_names
    compare_f1 = compare_f1 * 100
    compare_f1['epoch'] = compare_f1.index + 1
    compare_f1.index = compare_f1['epoch']
    compare_f1 = compare_f1.drop(columns=['epoch'])


    compare_f1.plot(y=column_names, kind='line', figsize=(20, 6))
    plt.xticks(compare_f1.index)
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title(f'Comparison of F1 Scores on {sample_set} Set')
    plt.legend(title='Data Sets')
    plt.show()
    
    return compare_f1

