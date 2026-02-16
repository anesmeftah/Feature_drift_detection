DATA_PATH = "data/raw/online_retail_II.csv"

HIGH_VALUE_PERCENTILE = 0.95
TRAIN_END_DATE = "2010-06-30"
VALIDATION_END_DATE = "2010-09-30"




REQUIRED_COLUMNS = [
    'Quantity',
    'InvoiceDate', 
    'Price', 
    'Country',
    'StockCode'
]

REQUIRED_CLEAN_COLUMNS = ['InvoiceDate', 'Country', 'StockCode', 'Year', 'Month', 'Day', 'Hour',
       'min', 'sec', 'is_weekend', 'day_of_week', 'hour_sin', 'hour_cos',
       'quarter', 'stockcode_freq', 'country_freq', 'target']

NUMERIC_FEATURES = [
    "Hour" , "Month" , "country_freq" , "stockcode_freq"
]

CATEGORICAL_FEATURES = [
    'Country',
    'StockCode',
    'InvoiceDate'
]

