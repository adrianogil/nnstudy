import nnexperiments.neural.neuralnetwork as nnet

nn = nnet.NeuralNetwork([2, 2, 1], [nnet.logsigmoid, nnet.logsigmoid])
