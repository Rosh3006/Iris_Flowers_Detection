import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("Iris.csv")
print(df.head())
df=df.drop(columns =['Id'])
df.head()
df.describe()
df.isnull().sum()
plt.subplots(figsize=(5,4))
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Species"]=le.fit_transform(df["Species"])
from sklearn.model_selection import train_test_split
X=df.drop(columns=['Species'])
Y=df["Species"]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x_train)
x_train= sc.transform(x_train)
x_test= sc.transform(x_test)
from sklearn.linear_model import LogisticRegression
lor=LogisticRegression()
lor.fit(x_train,y_train)
print("Accuracy:",lor.score(x_test,y_test)*100)
from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()
knc.fit(x_train,y_train)
print("Accuracy:",knc.score(x_test,y_test)*100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
print("Accuracy:",dt.score(x_test,y_test)*100)
import pickle
filename="saveModel.sav"
pickle.dump(knc,open(filename,'wb'))
load_model=pickle.load(open(filename,'rb'))
y=load_model.predict([[6.0,2.2,4.0,1.0]])
print(y)