import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

X, y = spiral_data(100, 3)

# plt.scatter(X[:,0], X[:,1])
# plt.show()

# plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
# plt.show()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
        
class Activation_ReLU:
    def forward(self, inputs):
        # if output <= 0: 0 else: output
        self.output = np.maximum(0, inputs)
    
layer1 = Layer_Dense(2,5)
activation = Activation_ReLU()

layer1.forward(X)
activation.forward(layer1.output)
print(activation.output)

# layer2.forward(layer1.output)
# print(layer2.output)
