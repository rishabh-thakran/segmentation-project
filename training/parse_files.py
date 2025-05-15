import pandas as pd
import os

def parse_test_file(file_path):
    """ Parses a test file to extract image names. """
    try:
        df = pd.read_csv(file_path, header=None, names=["image"])
        return df["image"].tolist()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def parse_train_val_file(file_path):
    """ Parses a train or validation file to extract image names and categories. """
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=["image", "category"])
        return df["image"].tolist(), df["category"].tolist()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [], []

def read_categories_file(file_path):
    """ Reads a categories file containing class names. """
    try:
        df = pd.read_csv(file_path, sep=r'\s+', engine='python')
        return df["Name"].tolist()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []
