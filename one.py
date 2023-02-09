
#import all the modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

#read data using read_csv
data = pd.read_csv("glass.csv")

#store all the data except last column in x , store last column means target data into y
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

#here we are spliting data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Train the Gaussian Naive Bayes model on the training data
g= GaussianNB()
g.fit(X_train, y_train)

#Predict the target variable for the test data using predict fun 
y_pred = g.predict(X_test)

#check accuracy  
print("Accuracy:", accuracy_score(y_test, y_pred))
#Evaluate the model on test part using score
print("Classification Report:")
"""
Precision: Precision is the number of true positive predictions divided by 
the number of true positive predictions plus false positive predictions.
Precision measures how well the model avoids false positives.

Recall (Sensitivity or True Positive Rate): Recall is the number of true
positive predictions divided by the number of true positive predictions plus false negative predictions.
Recall measures how well the model detects all positive instances.

f1-score: The F1 score is the harmonic mean of precision and recall.
The F1 score provides a balance between precision and recall.

Support: Support is the number of instances in the test data that belong to a particular class.

All of these metrics can help you to determine the overall performance 
of the classifier and identify areas where it may need improvement."""

print(classification_report(y_test, y_pred))