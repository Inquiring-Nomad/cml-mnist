import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
import numpy as np
# Set random seed
seed = 42


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

rf=RandomForestClassifier(n_estimators=100,random_state=seed)

train_score = cross_val_score(rf, X_train, y_train,cv=5)

predictions = cross_val_predict(rf, X_train, y_train,cv=5)


with open("metrics.txt", 'w') as outfile:
        outfile.write(f"Mean Cross Validation Score : {100 * np.mean(train_score):.2f}\n")





rf.fit(X_train, y_train)
# Test set
test_predictions = rf.predict(X_test)
test_acc = accuracy_score(y_test,test_predictions)
with open("metrics.txt", 'a') as outfile:
        outfile.write(f"Test Score : {100 * test_acc:.2f}\n")
skplt.metrics.plot_confusion_matrix(y_test, test_predictions, normalize=True)
plt.savefig("test_confusion_matrix.png")
