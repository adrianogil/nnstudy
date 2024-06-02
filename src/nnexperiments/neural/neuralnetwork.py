import numpy as np

try:
    xrange
except NameError:
    xrange = range

def hardlim(n):
    for x in np.nditer(n, op_flags=['readwrite']):
        if x[...] < 0:
            x[...] = 0
        else:
            x[...] =  1
    return n


def hardlims(n):
    for x in np.nditer(n, op_flags=['readwrite']):
        if x[...] < 0:
            x[...] = -1
        else:
            x[...] =  1
    return n

def purelin(n):
    return n

def satlin(n):
    for x in np.nditer(n, op_flags=['readwrite']):
        if x[...] < 0:
            x[...] = 0
        elif x[...] <= 1:
            x[...] = x[...]
        else:
            x[...] =  1
    return n

def satlins(n):
    for x in np.nditer(n, op_flags=['readwrite']):
        if x[...] < -1:
            x[...] = -1
        elif x[...] <= 1:
            x[...] = x[...]
        else:
            x[...] =  1
    return n

def logsigmoid(n, derivative=False):
    if derivative:
        sigmoid_result = logsigmoid(n)
        return sigmoid_result * (1 - sigmoid_result)

    for x in np.nditer(n, op_flags=['readwrite']):
        x[...] = 1.0 / (1.0 + np.exp(-x[...]))
    return n

def tansig(n):
    for x in np.nditer(n, op_flags=['readwrite']):
        e1 = np.exp(x[...])
        e2 = np.exp(-x[...])
        x[...] = (e1 + e2) / (e1 + e2)
    return n

def poslin(n):
    for x in np.nditer(n, op_flags=['readwrite']):
        if x[...] < 0:
            x[...] = 0
        else:
            x[...] = x[...]
    return n

class NeuralLayer:
    def __init__(self, input_size, output_size, neural_function):
        self.input_size = input_size
        self.output_size = output_size
        # self.neurons = neurons
        self.neural_function = neural_function
        self.weights = np.random.random((output_size, input_size))
        self.bias = np.random.random((output_size, 1))

    def get_output(self, input_vector):
        n = self.weights.dot(input_vector) + self.bias
        # print('debug: layer - n ' + str(n))
        return self.neural_function(n)

class NeuralNetwork:
    def __init__(self, arch, neural_functions):
        self.arch = arch
        self.layers = []

        for i in xrange(0, len(self.arch)-1):
            self.layers.append(NeuralLayer(self.arch[i],
                                           self.arch[i+1],
                                           neural_functions[i]))
    def get_output(self, input_vector):
        nn_output = input_vector

        for i in xrange(0, len(self.arch)-1):
            # print('debug: nn_input - ' + str(nn_output))
            nn_output = self.layers[i].get_output(nn_output)
            # print('debug: nn_output - ' + str(nn_output))

        return nn_output

    def get_weights(self):
        w = []

        for i in xrange(0, len(self.arch)-1):
            for lw in self.layers[i].weights:
                w.append(lw)

        return w

    def set_weights(self, weights):
        for i in range(0, len(self.arch)-1):
            self.layers[i].weights = weights[:len(self.layers[i].weights)]
            self.layers[i].weights =  self.layers[i].weights[len(self.layers[i].weights):]
