import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

numeric_features = ['Id', 'LotArea', 'YearBuilt',
                    'BsmtFinSF1', 'BedroomAbvGr',
                    'KitchenAbvGr', 'GarageArea', '1stFlrSF']

categorical_features = ['MSZoning', 'Heating']

target = 'SalePrice'
feature_list = ['Id', 'LotArea', 'YearBuilt', 'BsmtFinSF1', 'BedroomAbvGr',
                'KitchenAbvGr', 'GarageArea',
                '1stFlrSF', 'MSZoning', 'Heating']

model_scaler_1 = '../models/scaler.joblib'

model_encoder_1 = '../models/encoder.joblib'


# This function sieves the dataset for only useful features and target

def sieve_data(df):
    sieved_data = df[feature_list].join(df[target])
    return sieved_data


def drop_missing_rows(df):
    complete_rows = df.dropna()
    return complete_rows


def scale_numeric(df):
    scaler = StandardScaler()
    scaler.fit(df[numeric_features])
    scaled_data = scaler.transform(df[numeric_features])
    scaled_df = pd.DataFrame(data=scaled_data, columns=numeric_features)
    joblib.dump(scaler, open(model_scaler_1, 'wb'))
    return scaled_df


def encode_categorical(df):
    encoder = OrdinalEncoder()
    encoder.fit(df[categorical_features])
    encoded_data = encoder.transform(df[categorical_features])
    encoded_df = pd.DataFrame(data=encoded_data, columns=categorical_features)
    joblib.dump(encoder, open(model_encoder_1, 'wb'))
    return encoded_df


def hold_on_target(df):
    held_target = df[target]
    held_df = pd.DataFrame(held_target)
    return held_df


def comb_scal_enco(df1, df2, df3):
    merge_coders = df1.join(df2).join(df3)
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
    # holding taget in place
    output_hold = hold_on_target(output_drop)
    # Feature engineering by combining
    preprocessed_output = comb_scal_enco(
        output_scale, output_encode, output_hold)
    return preprocessed_output
