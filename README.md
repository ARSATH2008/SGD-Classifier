# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import and Load Dataset
   
2. Preprocess the Data
 
3. Train the Model
  
4. Test and Evaluate


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: J Mohamed Arsath
RegisterNumber:  25000358
*/
```
```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


iris=load_iris()

df=pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']= iris.target
print(df.head())

X=df.drop('target',axis=1)
y=df['target']

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=42)

sgd_clf=SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train,y_train)

y_pred =sgd_clf.predict(X_test)

accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm=confusion_matrix(y_test, y_pred)
print("confusion Matrix:")
print(cm)

report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:\n", report)
```

## Output:
![WhatsApp Image 2026-02-06 at 2 05 52 PM](https://github.com/user-attachments/assets/7757a303-46f6-444f-a145-5532c858b576)
![WhatsApp Image 2026-02-06 at 2 06 05 PM](https://github.com/user-attachments/assets/851d5391-0dd5-41dd-87ea-d186f47a848e)
![WhatsApp Image 2026-02-06 at 2 06 07 PM](https://github.com/user-attachments/assets/3267271a-9920-4290-a405-3a45e9508c81)







## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
