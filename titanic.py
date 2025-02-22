import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('C:\\Users\\arham\\OneDrive\\Desktop\\archive\\tested.csv')

# Check missing values
print(data.isnull().sum())

# Basic statistics
print(data.describe())

# Survival rate
survival_rate = data['Survived'].mean()
print(f"Survival Rate: {survival_rate * 100:.2f}%")

# Survival by gender
survival_by_gender = data.groupby('Sex')['Survived'].mean()
print(survival_by_gender)

# Survival by class
survival_by_class = data.groupby('Pclass')['Survived'].mean()
print(survival_by_class)

# Visualize survival by gender
sns.countplot(x='Sex', hue='Survived', data=data)
plt.show()

# Visualize survival by class
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.show()