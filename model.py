import numpy as np
import pandas as pd
import pickle 
from keras.models import model_from_json

dataset = pd.read_csv('kredi.csv',sep=";")

dataset.head()

X=dataset.iloc[:,0:5].values  
y=dataset.KrediDurumu.values.reshape(-1,1)

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

X[:, 2] = labelencoder.fit_transform(X[:, 2])
X[:, 4] = labelencoder.fit_transform(X[:, 4])

from sklearn import model_selection

validation_size = 0.20
seed = 5
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=12) # n_neigbors = k
model.fit(X_train,Y_train.ravel())

#filename = 'model.pkl'
#pickle.dump(knn, open(filename, "wb"))

#model = pickle.load(open("model.pkl", "rb"))
#print("tamamlandı.....")


model_json = model.to_json()
with open("knn_model.json", "w") as json_file:
    json_file.write(model_json)
    
model.save_weights("knn_model.h5")
print("Saved model to disk")

json_file = open('knn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("knn_model.h5")
print("Loaded model from disk")
