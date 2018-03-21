from neuralnetwork import NeuralNetwork
import neuralnetwork as nnet

import perceptron

nn = NeuralNetwork([2,1], [nnet.hardlim])

input_samples = [[[0],[0]], [[0],[2]], [[2],[1]], [[3],[2]]]
targets = [[0], [0], [1], [1]]

perceptron.learn(nn, input_samples, targets)