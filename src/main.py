from src.functions1 import load_and_clean_data, split_data
from src.functions2 import train_logistic_regression, train_random_forest, evaluate_model

# Load and clean data
data_filepath = 'data/game_user_churn.csv'
df = load_and_clean_data(data_filepath)

# Split data
X_train, X_test, y_train, y_test = split_data(df)

# Train models
logistic_model = train_logistic_regression(X_train, y_train)
rf_model = train_random_forest(X_train, y_train)

# Evaluate models
logistic_eval = evaluate_model(logistic_model, X_test, y_test)
rf_eval = evaluate_model(rf_model, X_test, y_test)

# Select best model (Random Forest as example)
best_model = rf_model if rf_eval['f1_score'] > logistic_eval['f1_score'] else logistic_model

# Save best model
import pickle
with open('src/models/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Best Model saved!")
