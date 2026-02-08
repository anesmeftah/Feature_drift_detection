import lightgbm as lgb
import optuna
from pathlib import Path

from data_loader import load_data
from features import add_time_features , create_target
import config
from sklearn.metrics import accuracy_score

CATEGORICAL_FEATURES = [
    'Country',
    'StockCode'
]

df = load_data("data/raw/online_retail_II.csv")

X = add_time_features(df)
y = create_target(df , 0.95)

train_mask = df["InvoiceDate"] <= config.TRAIN_END_DATE
val_mask   = (df["InvoiceDate"] > config.TRAIN_END_DATE) & (df["InvoiceDate"] <= config.VALIDATION_END_DATE)
test_mask  = df["InvoiceDate"] > config.VALIDATION_END_DATE

X = X.drop('InvoiceDate', axis=1)
X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask], y[val_mask]
X_test,  y_test  = X[test_mask], y[test_mask]

for col in CATEGORICAL_FEATURES:
    X_train[col] = X_train[col].astype('category')
    X_val[col] = X_val[col].astype('category')


enc = X_train['StockCode'].value_counts()
X_train['StockCode_Freq'] = X_train['StockCode'].map(enc)
X_val['StockCode_Freq'] = X_val['StockCode'].map(enc).fillna(0)

X_train.drop('StockCode', axis=1, inplace=True)
X_val.drop('StockCode', axis=1, inplace=True)

CATEGORICAL_FEATURES = [
    'Country'
]


dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
dvalid = lgb.Dataset(X_val, label=y_val, categorical_feature=CATEGORICAL_FEATURES, reference=dtrain)


def objective(trial):
    param = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "feature_pre_filter": False,
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True), 
        "device": "gpu"
    }

    gbm = lgb.train(
        param, 
        dtrain, 
        valid_sets=[dvalid],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=0) 
        ]
    )
    
    preds = gbm.predict(X_val)
    pred_labels = (preds >= 0.5).astype(int)
    accuracy = accuracy_score(y_val, pred_labels)
    return accuracy


study = optuna.create_study(direction="maximize")
study.optimize(objective , n_trials=10)

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


best_params = study.best_params
best_params.update({
    "objective": "binary",
    "metric": "binary_logloss",
    "verbose": -1,
    "boosting_type": "gbdt",
    "device": "gpu",
})

dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
gbm = lgb.train(best_params, dtrain)

gbm.save_model("../models/lgbm_optuna.txt")
