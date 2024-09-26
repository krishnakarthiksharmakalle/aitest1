import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_clean_data("E:\ai eng\game_user_churn (AI Eng).csv"):

    df = pd.read_csv("E:\ai eng\game_user_churn (AI Eng).csv")
    

    df.fillna(df.mean(), inplace=True)
    
    label_encoder = LabelEncoder()
    for column in ['gender', 'country', 'game_genre', 'subscription_status', 'device_type', 'favorite_game_mode']:
        df[column] = label_encoder.fit_transform(df[column].astype(str))
    
    return df

def split_data(df, target_column='churn'):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test
