import numpy as np

try:
    xrange
except NameError:
    xrange = range

def learn(nn, input_samples, targets, epoches=10, min_mse=1e-16, max_mse=1e+100):

    if len(nn.layers) > 1:
        print("Error: " + \
            "perceptron learning rule can be applied only in one layer architectures!")
        return

    if len(input_samples) != len(targets):
        print("Error: " + \
            "Input and Targets list should have the same size!")
        return

    for e in xrange(0, epoches):

        mse = 0
        for i in xrange(0, len(input_samples)):
            nn_output = nn.get_output(input_samples[i])
            print('For epoch %s and input %s got output %s given target %s' \
                % (e, i, nn_output, targets[i]))
            nn_error = targets[i] - nn_output
            mse = mse + np.inner(nn_error, nn_error)
            nn.layers[0].weights = nn.layers[0].weights + \
                    np.dot(nn_error,np.transpose(input_samples[i]))
            nn.layers[0].bias = nn.layers[0].bias + nn_error
        if mse < min_mse:
            print('At epoch %s training achieved MSE min limit: %s' %(e, mse))
            return
        if mse > max_mse:
            print('At epoch %s training achieved MSE max limit: %s' %(e, mse))
            return
        print('For epoch %s got MSE %s' %(e, mse))



