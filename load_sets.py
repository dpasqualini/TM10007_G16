import pandas as pd
import os


def load_ft_set():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    ft_usable = pd.read_csv(os.path.join(this_directory, 'Ft_set.csv'), index_col=0)

    return ft_usable


def load_d_set():
    this_directory = os.path.dirname(os.path.abspath(__file__))
    d_usable = pd.read_csv(os.path.join(this_directory, 'D_set.csv'), index_col=0)

    return d_usable
