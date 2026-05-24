import numpy as np

def sigmoid(x):
  """Sigmoid activation function."""
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  """Derivative of the sigmoid function."""
  return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, layer_sizes):
        """
        Initializes the neural network.

        Args:
            input_size: The number of features in the input data (6*5 = 30 in your case).
            layer_sizes: A list of integers representing the number of neurons in each hidden layer and the output layer.  e.g., [10, 5, 1] for two hidden layers with 10 and 5 neurons, and a single output neuron.
        """
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []

        # Initialize weights and biases randomly
        prev_size = input_size
        for size in layer_sizes:
            self.weights.append(np.random.randn(prev_size, size))
            self.biases.append(np.random.randn(1, size))
            prev_size = size

    def forward(self, X):
        """Performs a forward pass through the network."""
        a = X
        activations = [a]  # Keep track of activations for backpropagation

        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = sigmoid(z)
            activations.append(a)

        return activations

    def backward(self, X, y, activations, learning_rate):
        """Performs a backward pass and updates weights."""
        delta = activations[-1] - y  # Output layer error
        dw = []
        db = []

        for i in range(len(self.weights) - 1, -1, -1):
            dw_i = np.dot(activations[i].T, delta * sigmoid_derivative(activations[i+1]))
            db_i = np.sum(delta * sigmoid_derivative(activations[i+1]), axis=0, keepdims=True)
            dw.insert(0, dw_i)
            db.insert(0, db_i)
            delta = np.dot(delta, self.weights[i].T) * sigmoid_derivative(activations[i])

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dw[i]
            self.biases[i] -= learning_rate * db[i]


    def train(self, X, y, epochs, learning_rate):
        """Trains the neural network."""
        for epoch in range(epochs):
            activations = self.forward(X)
            self.backward(X, y, activations, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {np.mean((activations[-1] - y)**2)}") # Mean Squared Error


# Example usage:
input_size = 30  # 6 x 5 input array
layer_sizes = [10, 5, 1]  # Example: 10 neurons in 1st hidden layer, 5 in 2nd, 1 output neuron

nn = NeuralNetwork(input_size, layer_sizes)

# Sample data (replace with your actual data)
X = np.random.rand(100, input_size)  # 100 samples
y = np.random.rand(100, 1)  # 100 corresponding outputs

nn.train(X, y, epochs=1000, learning_rate=0.1)

#Prediction
predictions = nn.forward(X)[-1]
print("Predictions:", predictions)
