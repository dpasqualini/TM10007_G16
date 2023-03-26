import pandas as pd
import os


def load_Ft_set():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(this_directory)
    data_folder = os.path.join(root, 'datasets')
    Ft_usable = pd.read_csv(os.path.join(data_folder, 'Ft_set.csv'), index_col=0)

    return Ft_usable


def load_D_set():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(this_directory)
    data_folder = os.path.join(root, 'datasets')
    D_usable = pd.read_csv(os.path.join(data_folder, 'D_set.csv'), index_col=0)

    return D_usable
