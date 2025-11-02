#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ucimlrepo')


# In[2]:


from ucimlrepo import fetch_ucirepo


# In[3]:


heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# metadata
# print(heart_disease.metadata)

# variable information
# print(heart_disease.variables)


# In[4]:


import pandas as pd
from ucimlrepo import fetch_ucirepo
import numpy as np

heart_disease = fetch_ucirepo(id=45)


X = heart_disease.data.features
y = heart_disease.data.targets

dataset = pd.concat([X, y], axis=1)
dataset.head()


# In[5]:


dataset.info()


# In[6]:


dataset = dataset.replace({'num':{2:1, 3:1, 4:1}})


# In[7]:


# Check for non-integer values
non_integer_values = X.apply(pd.to_numeric, errors='coerce').isnull()

# Filter rows with non-integer values
rows_with_non_integer = X[non_integer_values.any(axis=1)]

print("Rows with non-integer values:")
print(rows_with_non_integer)


# In[8]:


# Impute missing 'ca' values with the mode (most frequent value)
X['ca'] = X['ca'].fillna(X['ca'].mode()[0])

# Impute missing 'thal' values with the mode (most frequent value)
X['thal'] = X['thal'].fillna(X['thal'].mode()[0])

print("DataFrame after imputation:")
print(X)


# In[9]:


# Check for non-integer values
non_integer_values = X.apply(pd.to_numeric, errors='coerce').isnull()

# Filter rows with non-integer values
rows_with_non_integer = X[non_integer_values.any(axis=1)]

print("Rows with non-integer values:")
print(rows_with_non_integer)


# In[10]:


y = y.replace({'num':{2:1, 3:1, 4:1}})


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[12]:


import matplotlib.pyplot as plt

# Example: Visualizing age and cholesterol
plt.scatter(X['age'], X['chol'], c=y['num'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('Cholesterol')
plt.title('Scatter Plot of Age vs Cholesterol')
plt.colorbar(label='Presence of Heart Disease')
plt.show()


# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the matplotlib figure and axes
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

# Plot age distribution
sns.histplot(data=X, x='age', bins=10, kde=True, ax=axes[0])
axes[0].set_title('Age Distribution')

# Plot sex distribution
sns.countplot(data=X, x='sex', ax=axes[1])
axes[1].set_title('Sex Distribution')
axes[1].set_xticklabels(['Female', 'Male'])

# Plot cholesterol level distribution
sns.histplot(data=X, x='chol', bins=10, kde=True, ax=axes[2])
axes[2].set_title('Cholesterol Level Distribution')

# Adjust layout
plt.tight_layout()
plt.show()


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

important_vars = ['age', 'chol','sex']

# Pairwise scatter plot matrix for important variables
sns.pairplot(X[important_vars])
plt.suptitle('Pairwise Relationships between Important Variables')
plt.show()


# In[15]:


# Correlation matrix
corr_matrix = X[['age', "trestbps", "chol", "thalach", "oldpeak", "ca"]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


# In[16]:


# Box plots
plt.figure(figsize=(10, 6))
sns.boxplot(data=X[['age', "trestbps", "chol", "thalach", "oldpeak"]])
plt.title('Box Plots of Variables')
plt.show()


# In[17]:


# X_without_outerliers = X.copy()
# for col in ["trestbps", "chol", "thalach", "oldpeak"]:
#   X_without_outerliers = replace_outliers(X_without_outerliers, col)


# In[18]:


X.describe()


# In[19]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# In[20]:


dataset['sex'] = dataset['sex'].replace({1: 'Male', 0: 'Female'})
dataset['num'] = dataset['num'].replace({1: 'Presence', 0: 'No Presence'})


# In[21]:


grouped = dataset.groupby('sex')['num'].value_counts(normalize=True).unstack(fill_value=0)

fig, axes = plt.subplots(1, grouped.shape[0], figsize=(10, 5))
for i, (sex, counts) in enumerate(grouped.iterrows()):
    axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'Sex: {sex}')

plt.tight_layout()
plt.show()


# In[22]:


grouped = dataset.groupby('ca')['num'].value_counts(normalize=True).unstack(fill_value=0)

fig, axes = plt.subplots(1, grouped.shape[0], figsize=(10, 5))
for i, (ca, counts) in enumerate(grouped.iterrows()):
    axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'Ca: {ca}')

plt.tight_layout()
plt.show()


# In[23]:


dataset['exang'] = dataset['exang'].replace({1: 'exercise induced angina: yes', 0: 'exercise induced angina: no'})
grouped = dataset.groupby('exang')['num'].value_counts(normalize=True).unstack(fill_value=0)

fig, axes = plt.subplots(1, grouped.shape[0], figsize=(10, 5))
for i, (exang, counts) in enumerate(grouped.iterrows()):
    axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'{exang}')

plt.tight_layout()
plt.show()


# In[24]:


dataset['thal'] = dataset['thal'].replace({3: 'normal', 6: 'fixed defect', 7: "reversable defect"})
grouped = dataset.groupby('thal')['num'].value_counts(normalize=True).unstack(fill_value=0)

fig, axes = plt.subplots(1, grouped.shape[0], figsize=(10, 5))
for i, (thal, counts) in enumerate(grouped.iterrows()):
    axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'{thal}')

plt.tight_layout()
plt.show()


# In[25]:


dataset['cp'] = dataset['cp'].replace({1: 'typical angina', 2: 'atypical angina', 3: "non-anginal pain", 4: "asymptomatic"})
grouped = dataset.groupby('cp')['num'].value_counts(normalize=True).unstack(fill_value=0)

fig, axes = plt.subplots(1, grouped.shape[0], figsize=(10, 5))
for i, (cp, counts) in enumerate(grouped.iterrows()):
    axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'{cp}')

plt.tight_layout()
plt.show()


# In[26]:


dataset['slope'] = dataset['slope'].replace({1: 'upsloping', 2: 'flat', 3: "downsloping"})
grouped = dataset.groupby('slope')['num'].value_counts(normalize=True).unstack(fill_value=0)

fig, axes = plt.subplots(1, grouped.shape[0], figsize=(10, 5))
for i, (slope, counts) in enumerate(grouped.iterrows()):
    axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'{slope}')

plt.tight_layout()
plt.show()


# In[27]:


sns.boxplot(x='num', y='chol', data=dataset)
plt.xlabel('Diseace')
plt.ylabel('serum cholestoral in mg/dl')
plt.show()


# In[28]:


sns.boxplot(x='num', y='oldpeak', data=dataset)
plt.xlabel('Diseace')
plt.ylabel('ST depression induced by exercise relative to rest')
plt.show()


# In[29]:


sns.boxplot(x='num', y='trestbps', data=dataset)
plt.xlabel('Diseace')
plt.ylabel('resting blood pressure')
plt.show()


# In[30]:


dataset['restecg'] = dataset['restecg'].replace({0: 'normal', 1: 'having ST-T wave abnormality', 2: "showing probable or definite left ventricular hypertrophy"})
grouped = dataset.groupby('restecg')['num'].value_counts(normalize=True).unstack(fill_value=0)

fig, axes = plt.subplots(1, grouped.shape[0], figsize=(10, 5))
for i, (restecg, counts) in enumerate(grouped.iterrows()):
    axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'{restecg}')

plt.tight_layout()
plt.show()


# In[31]:


sns.boxplot(x='num', y='age', data=dataset)
plt.xlabel('Diseace')
plt.ylabel('Age')
plt.show()


# In[32]:


sns.boxplot(x='num', y='thalach', data=dataset)
plt.xlabel('Diseace')
plt.ylabel('Amaximum heart rate achieved ge')
plt.show()


# In[33]:


dataset['fbs'] = dataset['fbs'].replace({0: 'fasting blood sugar < 120 mg/dl', 1: 'fasting blood sugar > 120 mg/dl'})
grouped = dataset.groupby('fbs')['num'].value_counts(normalize=True).unstack(fill_value=0)

fig, axes = plt.subplots(1, grouped.shape[0], figsize=(10, 5))
for i, (fbs, counts) in enumerate(grouped.iterrows()):
    axes[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    axes[i].set_title(f'{fbs}')

plt.tight_layout()
plt.show()


# ## Logestic Regression

# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[35]:


from sklearn.model_selection import train_test_split, GridSearchCV

categorical_features = ['sex', 'exang', "fbs"]
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10],
    'classifier__solver': ['liblinear']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, verbose=1, scoring='accuracy')

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# In[36]:


print(f'Accuracy: {accuracy}')


# In[37]:


import pandas as pd

best_pipeline = grid_search.best_estimator_
logreg = best_pipeline.named_steps['classifier']

coefficients = logreg.coef_[0]

num_features = best_pipeline.named_steps['preprocessor'].named_transformers_['num'].get_feature_names_out(numerical_features)

cat_features = best_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
cat_features = [f"{cat}_{val}" for cat in categorical_features for val in cat_features if val.startswith(cat + '_')]

feature_names = num_features.tolist() + cat_features

coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

print(coef_df.sort_values(by='Coefficient', ascending=False))


# In[38]:


import matplotlib.pyplot as plt
import seaborn as sns


coef_df['abs_Coefficient'] = coef_df['Coefficient'].abs()

coef_df = coef_df.sort_values(by='abs_Coefficient', ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x='abs_Coefficient', y='Feature', data=coef_df)
plt.title('Absolute Feature Importance in Logistic Regression Model')
plt.xlabel('Absolute Coefficient Value')
plt.ylabel('Features')
plt.show()


# In[39]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train the logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Predictions on the testing set
y_pred = lr_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Precision measures the proportion of true positive predictions out of all positive predictions made by the model. For class 0 (absence of heart disease), the precision is relatively high at 77%, indicating that when the model predicts the absence of heart disease, it's correct 77% of the time.
# Recall (also known as sensitivity) measures the proportion of true positive predictions out of all actual positive instances in the data. For class 1 (presence of heart disease), the recall is 33%, indicating that the model only correctly identifies 33% of the actual instances of heart disease.
# F1-score is the harmonic mean of precision and recall. It provides a balance between precision and recall. Class 0 has the highest F1-score of 0.84, indicating better performance for predicting absence of heart disease compared to other classes.
# Support indicates the number of actual occurrences of each class in the testing data.
# 
# Overall, while the model performs reasonably well in predicting the absence of heart disease (class 0), its performance for other classes, particularly for identifying the presence of heart disease, is limited. This suggests that further optimization of the model or exploration of other algorithms may be necessary to improve performance, especially for the minority classes.

# ## Random Forest

# In[40]:


categorical_features = ['sex', 'ca', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']


# In[41]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])


# In[42]:


from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


# In[67]:


rfc = RandomForestClassifier(random_state=42)


# In[68]:


param_grid_rfc = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}


# In[69]:


CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid_rfc, cv= 5)


# In[70]:


CV_rfc.fit(X_train, y_train)


# In[72]:


best_estimator_RFC = CV_rfc.best_estimator_
best_estimator_RFC


# In[73]:


y_pred_best_RFC = best_estimator_RFC.predict(X_test)


# In[74]:


accuracy_best_RFC = accuracy_score(y_test, y_pred_best_RFC)
print("Accuracy:", accuracy_best_RFC)
print("Classification Report:")
print(classification_report(y_test, y_pred_best_RFC))


# In[ ]:





# In[ ]:





# In[43]:


feature_names_num = numerical_features
feature_names_cat = preprocessor.named_transformers_['cat'].get_feature_names_out()

feature_names = feature_names_num + list(feature_names_cat)

importances = pipeline.named_steps['classifier'].feature_importances_

importances_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importances_df)


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importances_df)
plt.title('Feature Importances in RandomForest Model')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# In[45]:


from sklearn.ensemble import RandomForestClassifier

# Train a random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Feature importance scores
feature_importances = rf_model.feature_importances_
important_features = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
b = important_features.sort_values(by='Importance', ascending=False)

# Visualize feature importances
plt.figure(figsize=(10, 6))
sns.barplot(data=important_features, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance in Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

from sklearn.metrics import accuracy_score

# Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)




# In[46]:


important_features


# In[47]:


# Predictions on the testing set
y_pred_rf = rf_model.predict(X_test)

# Evaluate model performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", accuracy_rf)
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))


# ## SVC

# In[48]:


from sklearn.svm import SVC

categorical_features = ['sex', 'exang', "fbs"]
numerical_features = [col for col in X.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='linear'))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy:", accuracy)


# In[49]:


classifier = pipeline.named_steps['classifier']


feature_names_transformed = list(pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)) + numerical_features


coefficients = classifier.coef_[0]


import pandas as pd
coefficients_df = pd.DataFrame({
    'Feature': feature_names_transformed,
    'Coefficient': coefficients
})


coefficients_df = coefficients_df.sort_values(by='Coefficient', key=abs, ascending=False)


# In[50]:


coefficients_df['abs_Coefficient'] = coefficients_df['Coefficient'].abs()
plt.figure(figsize=(10, 8))
sns.barplot(x='abs_Coefficient', y='Feature', data=coefficients_df)
plt.title('Feature Importance in SVC Model')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.show()


# In[51]:


from sklearn.svm import SVC

# Train an SVM classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_svm = svm_model.predict(X_test)

# Calculate the accuracy of the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Print the accuracy
print("SVM Accuracy:", accuracy_svm)


# In[52]:


# List of the top 8 important features
important_features = ['thalach', 'oldpeak', 'chol', 'age', 'ca', 'trestbps', 'cp', 'thal']

# Select only the columns named in the list of important features
selected_df = X.loc[:, important_features]

# Display the selected DataFrame
print(selected_df)


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(selected_df, y, test_size=0.2, random_state=42)


# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Train the logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Predictions on the testing set
y_pred = lr_model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:





# In[55]:


from sklearn.preprocessing import StandardScaler

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler to the data and transform the data
X_scaled = scaler.fit_transform(X)




# In[56]:


from sklearn.cluster import KMeans

# Determine the optimal number of clusters (example using the elbow method)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Based on the elbow method, select the optimal number of clusters
# Example: Let's assume the optimal number of clusters is 3
k_optimal = 3

# Apply K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to the DataFrame
X['cluster'] = clusters

# Analyze the characteristics of each cluster
cluster_means = X.groupby('cluster').mean()
print(cluster_means)

# Visualize the clusters using scatter plots, box plots, or other visualization techniques
# You can also use dimensionality reduction techniques like PCA or t-SNE for visualization


# The elbow method is a technique used to determine the optimal number of clusters in a dataset for K-means clustering. It involves plotting the within-cluster sum of squares (inertia) as a function of the number of clusters and identifying the "elbow" point where the rate of decrease in inertia slows down significantly. This point often indicates the optimal number of clusters, We took 3 since after 3 no big change is there.
# 

# In[57]:


from sklearn.svm import SVC

# Train an SVM classifier
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_svm = svm_model.predict(X_test)

# Calculate the accuracy of the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Print the accuracy
print("SVM Accuracy:", accuracy_svm)


# In[ ]:





# In[58]:


Param_list = {'C': [0.1, 1, 10, 100],'kernel': ['linear', 'rbf'],'gamma':[1,0.1,0.001,0.0001]}


# In[ ]:





# In[59]:


SVC_model = SVC(random_state=42)

SVC_gridSearch = GridSearchCV(SVC_model,param_grid=Param_list,refit = True, verbose=2)


# In[60]:


SVC_gridSearch.fit(X_train, y_train)


# In[61]:


best_params = SVC_gridSearch.best_params_
best_params


# In[62]:


best_estimator = SVC_gridSearch.best_estimator_
best_estimator


# In[63]:


y_pred_best = best_estimator.predict(X_test)


# In[64]:


accuracy_best = accuracy_score(y_test, y_pred_best)


# In[65]:


print("Best SVM Accuracy:", accuracy_best)


# In[66]:


accuracy_best = accuracy_score(y_test, y_pred_best)
print("Accuracy:", accuracy_best)
print("Classification Report:")
print(classification_report(y_test, y_pred_best))

