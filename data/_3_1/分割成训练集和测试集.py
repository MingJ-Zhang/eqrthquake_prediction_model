import numpy as np
from sklearn.model_selection import train_test_split


data = np.load('./data.npy')

label = np.load('./label.npy')

label = np.reshape(label,[185,1])
print(label.shape)

x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

print(x_train.shape)

np.save('./x_train.npy',x_train)
np.save('./x_test.npy',x_test)
np.save('./y_train.npy',y_train)
np.save('./y_test.npy',y_test)
