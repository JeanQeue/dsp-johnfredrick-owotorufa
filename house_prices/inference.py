import numpy as np
import pandas as pd
import joblib

numeric_features = ['Id', 'LotArea', 'YearBuilt',
                    'BsmtFinSF1', 'BedroomAbvGr',
                    'KitchenAbvGr', 'GarageArea', '1stFlrSF']

categorical_features = ['MSZoning', 'Heating']

feature_list = ['Id', 'LotArea', 'YearBuilt', 'BsmtFinSF1', 'BedroomAbvGr',
                'KitchenAbvGr', 'GarageArea',
                '1stFlrSF', 'MSZoning', 'Heating']

model_scaler_1 = '../models/scaler.joblib'

model_encoder_1 = '../models/encoder.joblib'

model_store = '../models/model.joblib'

# This function sieves the dataset for only useful features and target


def sieve_data(df):
    sieved_data = df[feature_list]
    return sieved_data


def drop_missing_rows(df):
    complete_rows = df.dropna()
    return complete_rows


def scale_numeric(df):
    scaler = joblib.load(model_scaler_1)
    scaler.fit(df[numeric_features])
    scaled_data = scaler.transform(df[numeric_features])
    scaled_df = pd.DataFrame(data=scaled_data, columns=numeric_features)
    return scaled_df


def encode_categorical(df):
    encoder = joblib.load(model_encoder_1)
    encoder.fit(df[categorical_features])
    encoded_data = encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(data=encoded_data, columns=categorical_features)
    return encoded_df


def comb_scal_enco(df1, df2):
    merge_coders = df1.join(df2)
    return merge_coders


def preprocessing_step(df):
    # sieving
    output_sieve = sieve_data(df)
    # drop nan
    output_drop = drop_missing_rows(output_sieve)
    # scaling
    output_scale = scale_numeric(output_drop)
    # encode cat
    output_encode = encode_categorical(output_drop)
    # Feature engineering by combining
    preprocessed_output = comb_scal_enco(output_scale, output_encode)
    return preprocessed_output


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    # preprocessing dataframe
    preprocessed_data = preprocessing_step(input_data)

    # #Loading the model
    model = joblib.load(model_store)

    prediction = model.predict(preprocessed_data)

    return prediction
