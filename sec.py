#import sklearn and pandas module 
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

#using read_CSV file to read data
data = pd.read_csv("glass.csv")

#store all the data except last column in x , store last column means target data into y
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

##here we are spliting data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train a linear SVM model
create_lf = LinearSVC(random_state=0)
create_lf.fit(X_train, y_train)
#predict the test set 
y_pred = create_lf.predict(X_test)

# Evaluate the model
print("Test Accuracy:", create_lf.score(X_test, y_test))
print("Classification Report:")
print(classification_report(y_test, y_pred))