
 User Churn Prediction Project

Project Overview
This project was built from scratch to predict user churn in video games. The dataset used contains user-related data such as demographic information, gameplay habits, and social interactions. The main goal is to predict whether a user will churn, which means stopping or reducing game play within a given period. 

The solution includes:
- Data loading, cleaning, and preprocessing.
- Exploratory data analysis (EDA) to understand patterns in user churn.
- Feature engineering to enhance predictive power.
- Model building and training, using multiple models for comparison.
- Evaluation of models to select the best-performing one.
- Saving the trained model for future inference.

This project was implemented using Python without the use of Jupyter notebooks or auto-generated scripts. Every aspect of the solution, from data preprocessing to model evaluation, was coded from scratch.

---

 Directory Structure
```plaintext
user_churn_prediction/
│
├── data/
│   └── game_user_churn.csv              # Dataset
├── src/
│   ├── __init__.py                      # Marks 'src' as a package
│   ├── functions1.py                    # Data loading, preprocessing
│   ├── functions2.py                    # Model training, evaluation
│   ├── models/                          # Directory to store trained models
│       └── best_model.pkl               # Best-performing model saved here
├── main.py                              # Main script to execute the workflow
└── README.md                            # Project documentation
```

---

 How to Run the Project

 Prerequisites
To run this project, you will need Python 3.x and the following Python packages:
- `pandas`
- `scikit-learn`
- `seaborn`
- `matplotlib`
- `pickle`

You can install the required packages using the following command:
```bash
pip install pandas scikit-learn seaborn matplotlib
```

 Steps to Run:
1. Clone the Repository 
   Clone this project to your local machine.

   ```bash
   git clone <repository-url>
   cd user_churn_prediction
   ```

2. Place Dataset in the Data Folder  
   Make sure that the dataset file (`game_user_churn.csv`) is inside the `data/` folder. You should already have the file in the correct location if you followed the directory structure above.

3. Run the Main Script  
   Run the `main.py` file to start the process, which includes:
   - Loading and cleaning the data.
   - Splitting the data into training and testing sets.
   - Training two different models (Logistic Regression and Random Forest).
   - Evaluating both models and selecting the best one.
   - Saving the best model in the `models/` folder.

   Command:
   ```bash
   python main.py
   ```

4. Outputs 
   - The best-trained model will be saved in the `models/` directory as `best_model.pkl`.
   - You can extend the code to load this saved model for future predictions.

---

Workflow Breakdown

 1. Data Loading and Cleaning
The dataset is loaded using `pandas`. Missing values are handled appropriately, and categorical variables are label-encoded. Numerical columns are cleaned, and data is prepared for model training. The cleaning process ensures that the dataset is fit for training without any data inconsistencies.

2. Exploratory Data Analysis (EDA)
To gain insights into the dataset, we visualized several key relationships, including:
- The distribution of churned vs. non-churned users.
- How features like total play time, games played, and social interactions correlate with churn.

 3. Feature Engineering
New features were created to improve model performance, such as normalizing playtime data and encoding categorical features. Standard scaling was also applied to ensure uniformity among numerical features.

 4. Model Training
Two machine learning models were implemented:
- Logistic Regression: A linear model for binary classification.
- Random Forest: A tree-based ensemble model known for handling complex, nonlinear patterns.

Both models were trained on the processed dataset, and their performance was compared based on evaluation metrics like accuracy, precision, recall, and F1 score.

5. Model Evaluation
We evaluated the models on the test set and identified the best-performing model based on the F1 score. The results from each model were analyzed and compared to ensure robust predictions.

6. Feature Importance
Feature importance was analyzed for the Random Forest model. The most influential features for predicting churn included:
- Total play time
- Games played
- Number of social interactions

These insights help in understanding the behavior of churned users.

 7. Model Saving
The best-performing model was saved in the `models/` directory as `best_model.pkl`. This model can be reused for making predictions on new data.

 Project Highlights

- Built from Scratch: This project was entirely coded from scratch without relying on pre-built or automated solutions. Every line of code was crafted to handle the unique requirements of the task.
- Data Handling: Advanced data cleaning and feature engineering steps were performed to prepare the data optimally for machine learning models.
- Model Performance: A thorough comparison of multiple models was done to ensure the best possible prediction accuracy.
- Scalability: The project is designed to be easily scalable. New models, features, or datasets can be added with minimal changes to the existing structure.
- Clarity: The structure of the codebase ensures that it is easy to follow, with separate modules handling different parts of the workflow.

---

Generative AI Usage
Generative AI tools were used minimally and only for idea generation. No auto-generated code or direct outputs from AI tools were used in the final implementation. The entire pipeline and logic were developed manually to ensure full control over the solution.

---

Future Improvements
- More Models: Additional machine learning models (like Gradient Boosting, XGBoost) could be added to the pipeline for further improvements.
- Hyperparameter Tuning: Grid search and cross-validation could be implemented to fine-tune model parameters.
- Deployment: The model can be deployed as a web API for real-time predictions.



