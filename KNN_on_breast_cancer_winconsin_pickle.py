import numpy as np
from sklearn import preprocessing, cross_validation , neighbors
import pandas as pd
import pickle

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?' , -99999 ,inplace = True)
#df.dropna(inplace=True)

# IDK why df.dropna is not working and showing error

#now remove id column

df.drop(['id'] , 1  ,inplace =True)

X=np.array(df.drop(['class'] , 1))
y=np.array(df['class'])

#since the file is pickled we need not have to train 
'''
X_train , X_test , y_train , y_test =  cross_validation.train_test_split(X , y , test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train , y_train)
'''
# pickling
with open('K_nearest_pickle', 'wb') as k:
    pickle.dump(clf , k)
    
pickle_in =open('K_nearest_pickle' , 'rb')
clf=pickle.load(pickle_in)
#done pickling

accuracy =clf.score(X_test , y_test)
print(accuracy)
ex_msr=np.array([[4,2,1,1,1,2,3,2,1],[4,2,10,7,1,5,3,2,1]])
ex_msr=ex_msr.reshape(len(ex_msr),-1)
prediction = clf.predict(ex_msr)
print(prediction)