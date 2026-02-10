import pandas as pd
import numpy as np
from src import config

def add_time_features(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Year']  = df['InvoiceDate'].dt.year # type: ignore
    df['Month'] = df['InvoiceDate'].dt.month # type: ignore
    df['Day']   = df['InvoiceDate'].dt.day # type: ignore
    df['Hour']  = df['InvoiceDate'].dt.hour # type: ignore
    df['min'] = df['InvoiceDate'].dt.minute # type: ignore
    df['sec'] = df['InvoiceDate'].dt.second # type: ignore
    return df    


def create_target(df : pd.DataFrame , percentile : float) -> pd.Series:
    # Make the Series type explicit for static type checkers.
    total_amount: pd.Series = df['Quantity'].astype(float).mul(df['Price'].astype(float))
    threshold = float(total_amount.quantile(percentile))
    return (total_amount >= threshold).astype(int)\
    

