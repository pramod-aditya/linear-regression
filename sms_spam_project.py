# importing Modules

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# read csv file
data = pd.read_csv("spam.csv", encoding="latin_1")
print(data.head())

# find no.of rows and columns
print(data.shape)

# find if any NaN numbers is there in data
print(data.isnull().sum())

# Drop NAN Number columns
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# change column names
data = data.rename(columns={"v1": "label", "v2": "text"})

# Count observations in each label
print(data.label.value_counts())

# convert label to a numerical variable
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})
data['length'] = data['text'].apply(len)
print(data.head())

# importing the seaborn for plotting the graph
import seaborn as sns

sns.countplot(data["label"])
print(plt.show())
data["label"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.axis("equal")
print(plt.show())

# checking how many spam mails are there
spam1 = data.loc[data['label'] == 'spam']
print(spam1["text"].head())

# checking how many ham mails are there
ham1 = data.loc[data['label'] == 'ham']
print(ham1["text"].head())

# Assume x value as a input 
x = np.array(data.iloc[0:500, 1])
print(x[0:5])
print(x.shape)

# assume y value as target value
y = np.array(data.iloc[0:500, 0])
print(y[0:5])
print(y.shape)

# importing the Sklearn module
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# importing the CountVector from Feature extraction
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()
print(count_vector)

# apply count vector to convert text data vector format like 0s and 1s
train_data = count_vector.fit_transform(x_train)
test_data = count_vector.transform(x_test)

# importing the naive bayes algorithm form sklearn
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(train_data, y_train)

# find the prediction
pred = model.predict(test_data)
print(pred)

# checking accuracy
from sklearn.metrics import accuracy_score

score = accuracy_score(pred, y_test)
print(score)

# Testing
from sklearn.metrics import classification_report

nbreport = classification_report(y_test, pred)
print(nbreport)












