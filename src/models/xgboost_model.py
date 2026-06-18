import xgboost as xgb
import numpy as np

def fit_xgboost(X_train, y_train):
    dtrain = xgb.DMatrix(np.asarray(X_train), label=np.asarray(y_train))
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'verbosity': 0,
    }
    return xgb.train(params, dtrain, num_boost_round=100)

def predict_xgboost(model, X_test):
    if isinstance(model, xgb.Booster):
        return model.predict(xgb.DMatrix(np.asarray(X_test)))
    return model.predict(X_test)
