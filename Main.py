import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

X, y = spiral_data(100, 3)

'''  Dataset plot 
plt.scatter(X[:,0], X[:,1])
plt.show()

plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
plt.show()
'''

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
    
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probablities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probablities
        

dense1 = Layer_Dense(2,3)       
actition1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
actition2 = Activation_Softmax()
#  Layer 1 pass
dense1.forward(X)
actition1.forward(dense1.output)
# Layer 2 pass
dense2.forward(actition1.output)
actition2.forward(dense2.output)
# Output 
print(actition2.output[:5])


'''        This is the Activation_ReLU
layer1 = Layer_Dense(2,5)
activation = Activation_ReLU()

layer1.forward(X)
activation.forward(layer1.output)
print(activation.output)
'''
