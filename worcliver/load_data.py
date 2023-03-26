import pandas as pd
import os


def load_data():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    data = pd.read_csv(os.path.join(this_directory, 'Liver_radiomicFeatures.csv'), index_col=0)

    return data


def load_ft_set():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    Ft_usable = pd.read_csv(os.path.join(this_directory, 'Ft_set.csv'), index_col=0)

    return Ft_usable


def load_D_set(file_name):
    file_path = os.path.join(os.getcwd(), file_name)

    return pd.read_csv(file_path)
