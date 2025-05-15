import pandas as pd

def parse_test_file(file_path):
    try:
        df = pd.read_csv(file_path, header=None, names=["image"])
        return df["image"].tolist()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def parse_train_val_file(file_path):
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=["image", "category"])  # Updated here
        return df["image"].tolist(), df["category"].tolist()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [], []

def read_categories_file(file_path):
    try:
        df = pd.read_csv(file_path, sep=r'\s+', engine='python')  # Updated here
        return df["Name"].tolist()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
