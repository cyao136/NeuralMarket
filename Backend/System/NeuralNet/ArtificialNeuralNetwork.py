'''
Created on Oct 7, 2015

@author: Chengyu Yao
@summary: The learning class which incorporates the Artificial Neural Network model
'''

import numpy as np
from scipy.special import expit


class Connection(object):
    '''
    Connection
    '''
    def __init__(self, weight=None):
        if weight:
            self.weight = weight
        else:
            self.weight = np.random.random_sample()

class Neuron(object):
    '''
    Neuron
    '''
    def __init__(self, numOutput, layerIndex, outputValue=None, outputWeights=None):
        self.outputValue = outputValue
        self.outputWeights = outputWeights
        self.__create_connections(numOutput)
        self.__index = layerIndex

    def __create_connections(self, numOutput):
        for i in range(numOutput):
            self.outputWeights[i] = Connection()

    def forward_prop(self, prevLayer):
        sumOfValues = 0.0
        # Calculate the sum of the prevLayer's output values
        for neuron in prevLayer:
            sumOfValues += neuron.outputValue * neuron.outputWeights[self.__index].weight
            
        self.outputValue = self.activation_function(sumOfValues)
    
    def activation_function(self, value):
        return sigmoid(value)
    
    def activation_derivative(self, value):
        return expit(value)

class ArtificialNeuralNetwork(object):
    '''
    classdocs
    '''

    def __init__(self, layersSize=None):
        '''
        Constructor
        '''
        self.inputLayerSize = layersSize[0]
        self.outputLayerSize = layersSize[-1]
        self.__layers = []

        if layersSize:
            self.layersSize = layersSize
        else:
            self.layersSize = []

        # Create weight matrices for each layer to the next with random values
        #self.weights = []
        #self.__generate_weight_matrices()

        for i in range(len(self.layersSize)):
            numNeuronsNextLayer = 0
            if i < len(self.layersSize):
                numNeuronsNextLayer = self.layersSize[i+1]
            newLayer = self.__create_layer_with_neurons(self.layersSize[i], numNeuronsNextLayer)
            self.__layers[i] = newLayer

    def forward_prop(self, inputValues):
        '''
        Propagate input values forward in the network
        '''
        # Set output values for input layer
        for i in range(len(inputValues)):
            self.__layers[0][i].outputValue = inputValues[i]

        for layerNum in range(1, len(self.__layers)):
            prevLayer = self.__layers[layerNum-1]
            for neuronNum in range(len(self.__layers[layerNum])-1):
                self.__layers[layerNum][neuronNum].forward_prop(prevLayer)

    def back_prop(self, targetValues):
        '''
        Propagate target values back
        '''
        # Calculate overall net error

        # Calculate output layer gradients
        
        # Calculate hidden layer gradients
        
        # Update connection weights

    def get_results(self, resultValues):
        '''
        Get the result values
        '''

    def __create_layer_with_neurons(self, numNeurons, numNeuronsNextLayer, biasNeuronValue=1.0):
        '''
        Create a layer with numNeurons of Neurons
        '''
        newLayer = []
        # Add the neurons
        for index in range(numNeurons):
            newLayer[index] = Neuron(numNeuronsNextLayer, index)
        # Add the bias neuron
        newLayer.append(Neuron(numNeuronsNextLayer, outputValue=1.0))

        return newLayer
        
#===============================================================================
#     def __generate_weight_matrices(self):
# 
#         if self.hiddenLayerSize:
#             # Input to first hidden layer's weights
#             self.weights[0] = np.random.randn(self.inputLayerSize,
#                                               self.hiddenLayerSize[0])
# 
#             # Create weight matrices for the hidden layers
#             count = 0
#             while count < len(self.hiddenLayerSize)-1:
#                 newWeightMatrix = np.random.randn(self.hiddenLayerSize[count],
#                                                   self.hiddenLayerSize[count+1])
#                 self.weights.append(newWeightMatrix)
#                 count += 1
# 
#             # Last hidden layer to Output layer weight matrix
#             outputWeightMatrix = np.random.randn(self.hiddenLayerSize[count],
#                                                  self.outputLayerSize)
#             self.weights.append(outputWeightMatrix)
# 
#         else:
#             # Input to output weights directly
#             self.weights[0] = np.random.randn(self.inputLayerSize,
#                                               self.outputLayerSize)
#===============================================================================


if __name__ == "__main__":
    '''
    Main function
    '''

def sigmoid(self, x):
    return 1/(1+np.exp(-x))
