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

#Handling missing data
data['Age'].fillna(data['Age'].median(), inplace=True)
data.drop('Cabin', axis=1, inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

#Adding Features
data['FamilySize'] = data['SibSp'] + data['Parch']
data['Isalone'] = (data['FamilySize'] == 0).astype(int)
data['AgeGroup'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100], labels=['Child', 'Teen', 'Adult', 'Mid-Age', 'Senior'])
#Title
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')

# Visualize survival by gender
sns.countplot(x='Sex', hue='Survived', data=data)
plt.show()

# Visualize survival by class
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.show()

# Visualize survival by Familysize
sns.countplot(x='FamilySize', hue='Survived', data=data)
plt.title('Survival by Family Size')
plt.show()

# Visualize survival by Title
sns.countplot(x='Title', hue='Survived', data=data)
plt.title('Survival by Title')
plt.show()

# Visualize survival by Agegroup
sns.countplot(x='AgeGroup', hue='Survived', data=data)
plt.title('Survival by Age Group')
plt.show()

#BuildingModel
data = pd.get_dummies(data, columns=['Sex', 'Embarked', 'Title', 'AgeGroup'], drop_first=True)

from sklearn.model_selection import train_test_split
X = data.drop(['PassengerId', 'Name', 'Ticket', 'Survived'], axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")