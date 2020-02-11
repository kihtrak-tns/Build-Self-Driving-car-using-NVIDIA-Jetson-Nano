

# We import "Sequential" which is linear stacks of layers, because neural networks are composed on layers. For eg. the simple perceptron 
# model that we dealt before consisted of two layers, the i/p and the o/p. The "Dense" class denotes that every node in a layer is
# connected to the preceeding layer, forming a fully connected neural networks. 
# The optimizer we use is the Adam, which computes the adaptive learning rate for each parameter and handles big data sets efficiently.



import numpy as np
import matplotlib.pyplot as plt  
import keras
from keras.models import Sequential                                                
from keras.layers import Dense
from keras.optimizers import Adam


n_pts = 500
np.random.seed(0)
Xa = np.array([np.random.normal(13, 2, n_pts),
               np.random.normal(12, 2, n_pts)]).T
Xb = np.array([np.random.normal(8, 2, n_pts),
               np.random.normal(6, 2, n_pts)]).T
 
X = np.vstack((Xa, Xb))
y = np.matrix(np.append(np.zeros(n_pts), np.ones(n_pts))).T
 
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])

model = Sequential()
model.add(Dense(units = 1, input_shape = (2,), activation = 'sigmoid'))                              # Our model has one o/p and 2 i/p nodes
adam=Adam(lr = 0.1)

""" We need to configure the model - our model contains either 0 or 1. Hence we choose it to be a binary, if we deal categorical class entropy for our loss funtion that basically calculates the error. A metric is similar to error function, its value is to judge the model"""

model.compile(adam, loss = 'binary_crossentropy', metrics=['accuracy'])

""" We pass the labels to the points so that any misleading points w.r.t labels will be an error. Verbose is a progress bar, an epoch. An epoch denotes measurement of one iteration of the entire dataset and the number of epochs depends on the data size but more the epochs better the performance and more is the time to compute. More the epochs might also end in overfitting the data which is a modelling error when the function is too closely fit to a limited set of data points. But, this shouldn't be much of a problem with linear classifier with proper care. 
"Shuffle" -> as we tend to update weights in the direction of reducing the error, our function might get struck in a local minimum compared to the global minimum or absolute minimum. To combat this, in static data sets gradient descent sets get struck in local minima and the solution is to shuffle the training data. It shuffles the rows in data and trains them in subsets, so for every iteration it makes sure the algorithm bounces off the local minimum to the absolute minimum"""

h = model.fit(x=X, y=y, verbose=1, batch_size=50, epochs=50, shuffle='true')

plt.plot(h.history['acc'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy'])

# X -> The input data array (both the top and bottom ones)
# y -> Matrix of labels 
# model -> Sequential model that we created to train our data
# x_span & y_span -> Horizontal and vertical stretch from min and max values of X
## We add and subtract '1' to add tolerance in order to space out the values for better display
# xx, yy -> The .meshgrid() takes the 50 elements of both the arguments and convert it from 1D to 2D array by repeating the elements 
# in the row-wise (xx) and column-wise (yy).
# grid -> For every y coord we have 50 x coordinates, thereby forming a grid by concating flattened xx and yy.
# model.predict(grid), a keras function is going to gives us the predictions based on our input arrays
# z -> To ensure that our predictions array has the same size as our grid's coordinates
# .contourf() -> prints distinct contour zones. Each contour zone has an increased threshold of probability, to classify the data
# into distinct classes


def plot_decision_boundary(X, y, model) :
    x_span = np.linspace(min(X[:, 0]) - 1, max(X[:, 0]) + 1, 3)
    y_span = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1, 3)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    predictions_func = model.predict(grid)
    z = predictions_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)
    

    
# We check the prediction of the model without giving the labels to it.
plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])
 
plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])
x = 7.5
y = 5
 
 
point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="red")
print("prediction is: ",prediction)

