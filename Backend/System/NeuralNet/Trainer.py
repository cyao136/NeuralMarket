import os
import csv
import textwrap
from ArtificialNeuralNetwork import ArtificialNeuralNetwork


DATA_PATH = "Test/XOR.csv"
CYCLE = 5000


def train(topology, inputValues, targetValues):
    net = ArtificialNeuralNetwork(topology)
    count = 0
    threshold = None
    # When we expect the neural network to learn at earliest
    fastSuccess = 500
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
        # Find when it reached 95%+ accuracy
        if net.recentAverageError < 0.05 and not threshold:
            if count > fastSuccess:
                threshold = count
        count += 1
    
    print (textwrap.dedent("""\
        Stats:
            
            Reached 95% accuracy at: {}
        
    """.format(threshold)))
        
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
