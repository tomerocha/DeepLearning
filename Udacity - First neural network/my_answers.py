import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, inputs, targets):

        n_records = inputs.shape[0]
        delta_w_inputs_2_hidden = np.zeros(self.weights_input_to_hidden.shape)
        delta_w_hidden_2_output = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(inputs, targets):
            ### Forward pass ###
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)
            hidden_outputs = self.activation_function(hidden_inputs)

            # Output layer
            final_outputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

            ### Backward pass ###
            error = y - final_outputs

            # Calculate the hidden layer's contribution to the error
            hidden_error = np.dot(self.weights_hidden_to_output, error)

            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

            # Weight step (hidden to output)
            self.weights_hidden_to_output += self.lr * hidden_outputs[:, None] * error
            # Weight step (input to hidden)
            self.weights_input_to_hidden += self.lr * hidden_error_term * X[:, None]


    def run(self, features):
        # Hidden layer
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # Output layer
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 500
learning_rate = 0.1
hidden_nodes = 8
output_nodes = 1