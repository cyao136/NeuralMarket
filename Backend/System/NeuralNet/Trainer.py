import os
import csv
import textwrap
from ArtificialNeuralNetwork import ArtificialNeuralNetwork
import sys


DATA_PATH = "Test/XOR.csv"
CYCLE = 10000


def recordStats(count, thresholds, avgError, padding):

    if count > padding:
        # Find when it reached 95%+ accuracy
        if avgError < 0.05 and not thresholds[0]:
                thresholds[0] = count
        # Find when it reached 98%+ accuracy
        if avgError < 0.02 and not thresholds[1]:
                thresholds[1] = count
        # Find when it reached 99%+ accuracy
        if avgError < 0.01 and not thresholds[2]:
                thresholds[2] = count
        # Find when it reached 99.5%+ accuracy
        if avgError < 0.005 and not thresholds[3]:
                thresholds[3] = count


def userInput(network, inputValueNum):
    # Let the user input their data
    while True:
        inputValues = []
        for index in range(inputValueNum):
            val = input("Input {}: (quit to quit the program)  ".format(index + 1))
            if val == "quit":
                sys.exit(0)
            try:
                inputValues.append(float(val))
            except ValueError:
                print ("Not a float value. Try Again!\n")
                break
        print ("Input: {}".format(inputValues))
        network.forward_prop(inputValues)
        # get result
        result = network.get_results()
        print ("Result: {}".format(result))
        print ("---------------\n")


def train(topology, inputValues, targetValues):
    net = ArtificialNeuralNetwork(topology)
    count = 0
    # list of stats recording when it reached 95%, 98%, 99%, and 99.5%
    thresholds = [None, None, None, None]
    # When we expect the neural network to learn at earliest
    padding = 500
    while count < CYCLE:
        index = count % len(listInputValues)
        # propagate forward
        print ("Input: {}".format(listInputValues[index]))
        net.forward_prop(listInputValues[index])
        # get result
        result = net.get_results()
        print ("Result: {}".format(result))
        # back propagation
        print ("Target: {}".format(listTargetValues[index]))
        net.back_prop(listTargetValues[index])
        print ("Recent Average Error: {}".format(net.recentAverageError))
        print ("---------------\n")
        # record the stats
        recordStats(count, thresholds, net.recentAverageError, padding)
        count += 1
    userInput(net, topology[0])
    
    print (textwrap.dedent("""\
        Stats:
            
            Reached 95% accuracy at: {0}
            
            Reached 98% accuracy at: {1}
            
            Reached 99% accuracy at: {2}
            
            Reached 99.5% accuracy at: {3}
        
    """.format(thresholds[0], thresholds[1], thresholds[2], thresholds[3])))


if __name__ == "__main__":
    topology = []
    listInputValues = []
    listTargetValues = []
    with open(os.path.abspath(DATA_PATH), 'r') as file:
        reader = list(csv.reader(file))
        # first row is topology
        topology = list(map(int, reader[0]))
        for index in range(1, len(reader)):
            # Convert the list of strings into integers
            if index % 2:
                listInputValues.append(list(map(float, reader[index])))
            else:
                listTargetValues.append(list(map(float, reader[index])))
    if topology:
        print ("Topology: {}".format(topology))
        print ("Input: {}".format(listInputValues))
        print ("Target: {}".format(listTargetValues))
        train(topology, listInputValues, listTargetValues)
