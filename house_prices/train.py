import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error
from sklearn.ensemble import HistGradientBoostingRegressor
from house_prices.preprocess import preprocessing_step

target = 'SalePrice'
feature_list = ['Id', 'LotArea', 'YearBuilt', 'BsmtFinSF1', 'BedroomAbvGr',
                'KitchenAbvGr', 'GarageArea',
                '1stFlrSF', 'MSZoning', 'Heating']

model_store = '../models/model.joblib'


def training_data(preprocessed_df):
    # Splitting the data
    X, y = preprocessed_df.drop([target], axis=1), preprocessed_df[target]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Training the model
    hgb_regressor = HistGradientBoostingRegressor()
    trained_model = hgb_regressor.fit(X_train, y_train)
    # joblib.dump(hgb_regressor, model_store)
    return X_val, y_val, trained_model


def evaluation_trained_model(X_val, y_val, trained_model):
    # Predicting values for evaluation
    y_pred = trained_model.predict(X_val)

    # Evaluation metrics score
    rmsle = np.sqrt(mean_squared_log_error(y_val, y_pred))

    return {'RMSLE': rmsle}


def build_model(data: pd.DataFrame) -> dict[str, str]:
    # Preprocessing
    preprocessed_df = preprocessing_step(data)

    # Training
    X_val, y_val, trained_model = training_data(preprocessed_df)

    # Evaluation and result
    performance = evaluation_trained_model(X_val, y_val, trained_model)

    # save joblib
    joblib.dump(trained_model, open(model_store, 'wb'))

    return performance
