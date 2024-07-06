# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Display the first few rows of the dataset
print(df.head())

# Handling missing values
# Fill missing Age values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with the mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Dropping the Cabin column because it has too many missing values
df.drop(columns=['Cabin'], inplace=True)

# Dropping rows with missing values in the 'Fare' column
df.dropna(subset=['Fare'], inplace=True)

# Feature Engineering
# Creating a new feature 'FamilySize'
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Encoding categorical variables
# Encoding 'Sex' with LabelEncoder
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])

# One-hot encoding 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Normalizing numerical features
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# Dropping unnecessary columns
df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)

# Display the preprocessed dataset
print(df.head())
