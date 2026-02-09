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

NUMERIC_FEATURES = [
    'Quantity',
    'Price'
]

CATEGORICAL_FEATURES = [
    'Country',
    'StockCode',
    'InvoiceDate'
]

