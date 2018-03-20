from neuralnetwork import NeuralNetwork
import neuralnetwork as nnet

import perceptron

nn = NeuralNetwork([1,2,1], [nnet.hardlim, nnet.hardlim])

input_samples = [[[0],[0]], [[0],[1]], [[2],[0]], [[0],[3]]]
targets = [[0], [0], [1], [1]]

perceptron.learn(nn, input_samples, targets)