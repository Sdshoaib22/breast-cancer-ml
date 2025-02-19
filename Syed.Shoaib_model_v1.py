#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas numpy scikit-learn matplotlib jupyter


# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[3]:


# Step 1: Load the Breast Cancer Wisconsin Dataset
data = load_breast_cancer()

# Convert to DataFrame for easier handling
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Display first 5 rows
print("Dataset Head:")
print(df.head())


# In[4]:


# Step 2: Exploratory Data Analysis (EDA)
print("\nDataset Info:")
print(df.info())

print("\nClass Distribution:")
print(df['target'].value_counts())

# Visualize the class distribution
sns.countplot(x='target', data=df)
plt.title('Class Distribution (0 = Malignant, 1 = Benign)')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()


# In[5]:


# Step 3: Preprocess the Data
X = df.drop('target', axis=1)
y = df['target']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[6]:


# Step 4: Train the Logistic Regression Model
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)


# In[7]:


# Step 5: Evaluate the Model
y_pred = model.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=data.target_names))


# In[9]:


# Step 6: Save the Model as a Python Script
with open('syed.shoaib_model_v1.py', 'w') as f:
    f.write('''import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=10000)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
''')

print("\nModel saved as 'Syed.Shoaib_model_v1.py'")


# In[ ]:




