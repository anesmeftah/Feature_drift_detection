import pandas as pd
import numpy as np
import config

def load_data(path = config.DATA_PATH ) -> pd.DataFrame:
    temp_df = pd.read_csv(path)
    missing = set(config.REQUIRED_COLUMNS) - set(temp_df.columns)
    if missing : 
        raise ValueError(f"Missing required column : {missing}")
    
    df = temp_df[config.REQUIRED_COLUMNS]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


