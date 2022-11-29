# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import sys

# dataset = sys.path('diabetes.csv')
names = ['melahirkan', 'glukosa', 'darah', 'kulit', 'insulin', 'bmi', 'riwayat', 'umur', 'class']
dataframe = pandas.read_csv('D:\diabetes.csv', names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
X_train = np.delete(X_train, [2, 4], axis=1)
X_test =  np.delete(X_test, [2, 4], axis=1)
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'LG_model2.sav'
pickle.dump(model, open(filename, 'wb'))
# load the model from disk
loaded_model = pickle.load(open('LG_model2.sav', 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)
