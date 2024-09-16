from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with ZipFile(r'C:\Users\Waleed\Documents\GitHub\Nueral-Network\archive.zip') as zObject:
 ZipFile.extractall(path=r'C:\Users\Waleed\Documents\GitHub\Nueral-Network')
data = pd.read_csv(r'C:\Users\Waleed\Documents\GitHub\Nueral-Network\archive\mnist_train.csv')
test = pd.read_csv(r'C:\Users\Waleed\Documents\GitHub\Nueral-Network\archive\mnist_test.csv')
data = np.array(data)
test = np.array(test)
m, n = data.shape
np.random.shuffle(data)
data = data.T
test = test.T
Y_train = data[0]
X_train = data[1:n]
X_train = X_train / 255.

Y_test = test[0]
X_test = test[1:n]
X_test = X_test / 255.

_,m_train = X_train.shape

def softmax(Z):
    Z -= np.max(Z, axis=0) 
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_deriv(z):
    return sigmoid(z)*(1-sigmoid(z)) 

def init_params():
    W1 = np.random.normal(size=(20, 784)) * np.sqrt(1./(784))
    b1 = np.random.normal(size=(20, 1)) * np.sqrt(1./10)
    W2 = np.random.normal(size=(10, 20)) * np.sqrt(1./30)
    b2 = np.random.normal(size=(10, 1)) * np.sqrt(1./(784))
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * sigmoid_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations+1):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("accuracy=",get_accuracy(predictions, Y)*100,"%")
    return W1, b1, W2, b2

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 3 , 100)    
test = make_predictions(X_test, W1, b1, W2, b2)
# for i in range (10):
#     test_prediction(i,W1,b1,W2,b2)
print("test accuracy=",get_accuracy(test, Y_test)*100,"%")