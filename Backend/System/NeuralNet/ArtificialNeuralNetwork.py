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
        
        self.deltaWeight = 0.0

class Neuron(object):
    '''
    Neuron
    '''
    def __init__(self, numOutput, layerIndex, outputValue=None, outputWeights=None, rate=None, momentum=None):
        self.outputValue = outputValue
        self.outputWeights = outputWeights
        self.__create_connections(numOutput)
        self.__index = layerIndex
        self.gradient = None
        if rate:
            self.rate = rate
        else:
            self.rate = 0.2
        if momentum:
            self.momentum = momentum
        else:
            self.momentum = 0.5

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
    
    def calculate_output_gradients(self, targetValue):
        delta = targetValue - self.outputValue
        self.gradient = delta * self.activation_derivative(self.outputValue)
    
    def calculate_hidden_gradients(self, nextLayer):
        sumDOW = 0.0
        for index in range(len(nextLayer)-1):
            sumDOW += self.outputWeights[index].weight * nextLayer[index].gradient
        self.gradient = self.activation_derivative(sumDOW * self.outputValue)
        
    def update_input_weights(self, prevLayer):
        for index in range(len(prevLayer)):
            neuron = prevLayer[index]
            oldDeltaWeight = neuron.outputWeights[self.__index].deltaWeight
            newDeltaWeight = self.rate * neuron.outputValue * self.gradient + self.momentum * oldDeltaWeight
            neuron.outputWeights[self.__index].deltaWeight = newDeltaWeight
            neuron.outputWeights[self.__index].weight += newDeltaWeight

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
        self.__errorValue = 0.0
        self.__recentAverageError = 0.0
        self.__recentErrorFactor = 0.0

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
        outputLayer = self.__layers[-1]
        
        for index in range(len(outputLayer) - 1):
            delta = targetValues[index] - outputLayer[index].outputValue
            self.__errorValue += np.square(delta)
        self.__errorValue /= len(outputLayer) - 1
        self.__errorValue = np.sqrt(self.__errorValue)
        
        self.__recentAverageError =\
            (self.__recentAverageError * self.__recentErrorFactor + self.__errorValue)/(self.__recentErrorFactor + 1.0)
        
        # Calculate output layer gradients
        for index in range(len(outputLayer) - 1):
            outputLayer[index].calculate_output_gradients(targetValues[index])
        
        # Calculate hidden layer gradients
        hiddenLayerIndex = len(self.__layers) - 2
        while hiddenLayerIndex > 0:
            curLayer = hiddenLayers[hiddenLayerIndex]
            nextLayer = hiddenLayers[hiddenLayerIndex + 1]
            for neuron in curLayer:
                neuron.calculate_hidden_gradients(nextLayer)
            hiddenLayerIndex -= 1
        
        # Update connection weights
        layerIndex = len(self.__layers) - 1
        while layerIndex > 0:
            curLayer = self.__layers[layerIndex]
            prevLayer = self.__layers[layerIndex - 1]
            for neuron in curLayer:
                neuron.update_input_weights(prevLayer)
            layerIndex -= 1

    def get_results(self, resultValues):
        '''
        Get the result values
        '''
        resultValues = []
        for neuron in self.__layers[-1]:
            resultValues.append(neuron.outputValue)

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
        
        
if __name__ == "__main__":
    '''
    Main function
    '''

def sigmoid(self, x):
    return 1/(1+np.exp(-x))
