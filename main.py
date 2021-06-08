import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Data Providing start =======================================================
# Get Data from CSV
df=pd.read_csv("student_db.csv")
inputs=df.drop('dropout', axis='columns')
target=df['dropout']
# Split Data for test 
x_train,x_test,y_train,y_test= train_test_split(inputs,target,test_size=0.2)
# Data Providing End =======================================================

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(x_train, y_train)

# [1 = sem 1 result][ 2 = sem 2 result][ 3 = financial condition(2=good,1=midium,3=poor)][4 = mentoring happen 1 and not 0 ]
inputdata=[[52,80,1,0]]
y_pred=clf.predict(inputdata)
if y_pred==1:
    print("DROP COLLEGE")
else:
    print("continue... study")