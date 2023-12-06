import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('/content/HepatitisCdata.csv')
data
data.shape
data.info()
data.head()
data.tail()
data.tail()
data.describe()
data["Sex"].value_counts()
data["Unnamed: 0"].value_counts()
data["Category"].value_counts()
data["Age"].value_counts()
data["ALP"].value_counts()
data["ALB"].value_counts()
data["ALT"].value_counts()
data["AST"].value_counts()
data["BIL"].value_counts()
data["CHE"].value_counts()
data["CHOL"].value_counts()
data["CREA"].value_counts()
data["GGT"].value_counts()
data["PROT"].value_counts()
print('The highest unnamed:0 was of:',data['Unnamed: 0'].max())
print('The lowest unnamed:0 was of:',data['Unnamed: 0'].min())
print('The average unamed:0 in the data:',data['Unnamed: 0'].mean())
print('The highest age was of:',data['Age'].max())
print('The lowest age was of:',data['Age'].min())
print('The average age in the data:',data['Age'].mean())
print('The highest alp was of:',data['ALP'].max())
print('The lowest alp was of:',data['ALP'].min())
print('The average alp in the data:',data['ALP'].mean())
print('The highest alb was of:',data['ALB'].max())
print('The lowest alb was of:',data['ALB'].min())
print('The average alb in the data:',data['ALB'].mean())
print('The highest alt was of:',data['ALT'].max())
print('The lowest alt was of:',data['ALT'].min())
print('The average alt in the data:',data['ALT'].mean())
print('The highest ast was of:',data['AST'].max())
print('The lowest ast was of:',data['AST'].min())
print('The average ast in the data:',data['AST'].mean())
print('The highest bil was of:',data['BIL'].max())
print('The lowest bil was of:',data['BIL'].min())
print('The average bil in the data:',data['BIL'].mean())
print('The highest che was of:',data['CHE'].max())
print('The lowest che was of:',data['CHE'].min())
print('The average che in the data:',data['CHE'].mean())
print('The highest chol was of:',data['CHOL'].max())
print('The lowest chol was of:',data['CHOL'].min())
print('The average chol in the data:',data['CHOL'].mean())
print('The highest crea was of:',data['CREA'].max())
print('The lowest crea was of:',data['CREA'].min())
print('The average crea in the data:',data['CREA'].mean())
print('The highest ggt was of:',data['GGT'].max())
print('The lowest ggt was of:',data['GGT'].min())
print('The average ggt in the data:',data['GGT'].mean())
print('The highest prot was of:',data['PROT'].max())
print('The lowest prot was of:',data['PROT'].min())
print('The average prot in the data:',data['PROT'].mean())
plt.plot(data['CREA'])
plt.xlabel("CREA")
plt.ylabel("Levels")
plt.title("Line Plot")
plt.show()
sns.histplot(data['Category'])
plt.xticks(rotation="vertical")
plt.show()
for i in data.columns:
    sns.histplot(data[i])
    print('columns : ' , i )
    plt.xticks(rotation = 'vertical')
    plt.show()
from sklearn.preprocessing import LabelEncoder
# Assuming 'df' is your DataFrame and 'target_column' is the name of your target variable column
target_column = 'Category'
# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Fit and transform the target variable
data[target_column] = label_encoder.fit_transform(data[target_column])
data
from sklearn.preprocessing import LabelEncoder
# Assuming 'df' is your DataFrame and 'target_column' is the name of your target variable column
target_column = 'Category'
# Initialize LabelEncoder
label_encoder = LabelEncoder()
# Fit and transform the target variable
data[target_column] = label_encoder.fit_transform(data[target_column])
data[5:11]
data.isnull().sum()
data=data.fillna(method='bfill')
data.isnull().sum()
from sklearn import preprocessing
import pandas as pd
data.iloc[:, 1:11] = data.iloc[:, 1:11].apply(pd.to_numeric, errors='coerce').fillna(0)
d = preprocessing.normalize(data.iloc[:,1:11], axis=0)
scaled_df = pd.DataFrame(d, columns=[ "ALB","ALP","ALT","AST","BIL","CHE","CHOL","CREA","GGT","PROT"])
scaled_df.head()
import pandas as pd
# Assuming 'df' is your DataFrame
feature_names = data.columns.tolist()
data
feature_names
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
# Assuming 'classification' is a variable containing the target column name
classification = 'Category'  # Replace with your actual target column name
# Select features (X) and target variable (y)
feature_columns = ['Unnamed: 0',
 'Age',
 'Sex',
 'ALB',
 'ALP',
 'ALT',
 'AST',
 'BIL',
 'CHE',
 'CHOL',
 'CREA',
 'GGT',
 'PROT']
X = data[feature_columns]
y = data[classification]
# Replace '\t?' with NaN
X.replace('\t?', np.nan, inplace=True)
# Convert columns to numeric (assuming that they are numeric features)
X = X.apply(pd.to_numeric, errors='coerce')
# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
# Split the data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
# Initialize and train the Logistic Regression model
model = LogisticRegression()
model.fit(train_X, train_Y)
# Make predictions on the test set
predictions = model.predict(test_X)
# Evaluate the model
accuracy = metrics.accuracy_score(predictions, test_Y)
print('The accuracy of the Logistic Regression model is:', accuracy)
# Display the classification report
report = classification_report(test_Y, predictions)
print("Classification Report:\n", report)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# Assuming 'target' is a variable containing the target column name for regression
target = 'Category'  # Replace with your actual target column name
# Select features (X) and target variable (y)
feature_columns = ['Unnamed: 0',
 'Age',
 'Sex',
 'ALB',
 'ALP',
 'ALT',
 'AST',
 'BIL',
 'CHE',
 'CHOL',
 'CREA',
 'GGT',
 'PROT']
X = data[feature_columns]
y = data[target]
# Impute missing values using the mean strategy
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
# Split the data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(train_X, train_y)
# Make predictions on the test set
predictions = model.predict(test_X)
# Evaluate the model
mse = metrics.mean_squared_error(predictions, test_y)
print('Mean Squared Error of the Linear Regression model is:', mse)
# You might want to print other regression metrics depending on your needs
# hepatitis
