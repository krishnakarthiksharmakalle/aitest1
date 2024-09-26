# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance

# Step 1: Data Loading and Cleaning
def load_and_clean_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    for column in ['gender', 'country', 'game_genre', 'subscription_status', 'device_type', 'favorite_game_mode']:
        df[column] = label_encoder.fit_transform(df[column].astype(str))
    
    return df

# Step 2: Exploratory Data Analysis (EDA)
def exploratory_data_analysis(df):
    # Basic statistics
    print(df.describe())
    
    # Churn distribution
    sns.countplot(x='churn', data=df)
    plt.title('Churn Distribution')
    plt.show()
    
    # Visualize relationships between features and churn
    sns.boxplot(x='churn', y='total_play_time', data=df)
    plt.title('Total Play Time vs Churn')
    plt.show()

    sns.boxplot(x='churn', y='avg_session_time', data=df)
    plt.title('Average Session Time vs Churn')
    plt.show()

# Step 3: Feature Engineering
def feature_engineering(df):
    # Normalize numerical columns
    scaler = StandardScaler()
    numerical_columns = ['total_play_time', 'avg_session_time', 'games_played', 'in_game_purchases', 
                         'last_login', 'friend_count', 'daily_play_time', 'number_of_sessions', 'social_interactions', 'achievement_points']
    
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    return df

# Step 4: Train-Test Split
def split_data(df):
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Step 5: Model Training
def train_models(X_train, y_train):
    # Logistic Regression
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    
    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    
    return logistic_model, rf_model

# Step 6: Model Evaluation
def evaluate_models(models, X_test, y_test):
    evaluations = {}
    
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        evaluations[model_name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        }
    
    return evaluations

# Step 7: Feature Importance for Random Forest
def feature_importance(model, X_train, df):
    importance = model.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance for Random Forest')
    plt.show()

# Main function to run all steps
def main():
    # Load and clean data
    df = load_and_clean_data('game_user_churn.csv')
    
    # Perform EDA
    exploratory_data_analysis(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train models
    logistic_model, rf_model = train_models(X_train, y_train)
    
    # Evaluate models
    models = {'Logistic Regression': logistic_model, 'Random Forest': rf_model}
    evaluations = evaluate_models(models, X_test, y_test)
    
    # Print evaluation metrics
    for model_name, metrics in evaluations.items():
        print(f"{model_name} Performance:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        print()
    
    # Feature importance for Random Forest
    feature_importance(rf_model, X_train, df)

# Run the main function
if __name__ == "__main__":
    main()
