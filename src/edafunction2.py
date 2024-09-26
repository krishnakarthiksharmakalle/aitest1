import matplotlib.pyplot as plt
import seaborn as sns

def plot_churn_distribution(df):
    sns.countplot(x='churn', data=df)
    plt.title('Churn Distribution')
    plt.show()

def plot_feature_relationships(df):
    # Example: Plot total play time vs churn
    sns.boxplot(x='churn', y='total_play_time', data=df)
    plt.title('Total Play Time vs Churn')
    plt.show()
