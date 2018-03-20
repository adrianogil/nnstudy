import numpy as np

class NeuralLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        # self.neurons = neurons
        self.weights = np.random.random((output_size, input_size))

class NeuralNetwork:
    def __init__(self, arch):
        self.arch = arch
        self.layers = []

        for i in xrange(0, len(self.arch)-1):
            self.layers.append(NeuralLayer(self.arch[i], self.arch[i+1]))