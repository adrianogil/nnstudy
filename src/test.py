from neuralnetwork import NeuralNetwork

import perceptron

nn = NeuralNetwork([1,2,1])

input_samples = [[[0],[0]], [[0],[1]], [[2],[0]], [[0],[3]]]
targets = [[0], [0], [1], [1]]

perceptron.learn(nn, input_samples, targets)