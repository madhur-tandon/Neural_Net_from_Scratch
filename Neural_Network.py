import pickle
import numpy as np

def load_model(name):
  pickle_off = open(name,"rb")
  model = pickle.load(pickle_off)
  return model

np.random.seed(42)

class NeuralNetwork:
  def __init__(self, num_layers, nodes_in_each_layer, activation, learning_rate):
    self.num_layers = num_layers
    self.nodes_in_each_layer = nodes_in_each_layer
    self.activation = activation
    self.learning_rate = learning_rate
    self._create_architecture()

  @staticmethod
  def _accuracy(predicted_labels, test_labels):
    return np.mean(predicted_labels == test_labels)

  @staticmethod
  def _sigmoid(data):
    return 1 / (1 + np.exp(-1*data))

  @staticmethod
  def _tanh(data):
    pos = np.exp(data)
    neg = np.exp(-1*data)
    num = pos - neg
    den = pos + neg
    return num / den

  def _tanh_gradient(self, data):
    return np.multiply(1 - self._tanh(data), 1 + self._tanh(data))

  def _sigmoid_gradient(self, data):
    return np.multiply(self._sigmoid(data), 1 - self._sigmoid(data))

  @staticmethod
  def _relu(data):
    return np.where(data>0,data,0)

  @staticmethod
  def _softmax(data):
    max_vector = np.max(data,axis=1)[:,np.newaxis]
    data = data - max_vector
    denom = np.sum(np.exp(data), axis=1)[:,np.newaxis]
    return np.exp(data) / denom

  @staticmethod
  def _cross_entropy_loss(softmax_output, input_labels):
    loss = np.log(softmax_output[np.arange(len(input_labels)), input_labels])
    num_samples = input_labels.shape[0]
    return (-1*np.sum(loss) / num_samples)

  @staticmethod
  def _cross_entropy_gradient(softmax_output, input_labels):
    softmax_output[np.arange(len(input_labels)), input_labels] -= 1
    num_samples = input_labels.shape[0]
    gradients = softmax_output / num_samples
    return gradients

  def _create_architecture(self):
    self.all_weights = []
    self.all_biases = []

    for i in range(self.num_layers-1):
      weight_matrix = 0.01*np.random.randn(self.nodes_in_each_layer[i], self.nodes_in_each_layer[i+1])
      bias_vector = np.zeros((1, self.nodes_in_each_layer[i+1]))
      self.all_weights.append(weight_matrix)
      self.all_biases.append(bias_vector)

  def _forward(self, input_data):
    self.all_preactivations = []
    self.all_activations = []

    for i in range(len(self.all_weights)):
      if i!= len(self.all_weights) - 1:
        intermediate_output = np.dot(input_data, self.all_weights[i]) + self.all_biases[i]
        self.all_preactivations.append(intermediate_output)
        if self.activation=='sigmoid':
          activated_output = self._sigmoid(intermediate_output)
        elif self.activation=='relu':
          activated_output = self._relu(intermediate_output)
        elif self.activation=='tanh':
          activated_output = self._tanh(intermediate_output)
        elif self.activation=='linear':
          activated_output = intermediate_output
        input_data = activated_output
        self.all_activations.append(activated_output)
      else:
        intermediate_output = np.dot(input_data, self.all_weights[i]) + self.all_biases[i]
        activated_output = self._softmax(intermediate_output)

    softmax_output = activated_output
    return softmax_output

  def _backward(self, softmax_output, input_data, input_labels):
    self.all_dWs = [None for i in range(len(self.all_weights))]
    self.all_dBs = [None for i in range(len(self.all_weights))]
    outer_layer_gradients = self._cross_entropy_gradient(softmax_output, input_labels)
    dhidden = outer_layer_gradients
    for i in range(len(self.all_weights)-1,-1,-1):
      if i!=0:
        self.all_dWs[i] = np.dot(self.all_activations[i-1].T,dhidden)
        self.all_dBs[i] = np.sum(dhidden, axis=0)
        dhidden = np.dot(dhidden, self.all_weights[i].T)
        if self.activation=='sigmoid':
          dhidden = np.multiply(dhidden, self._sigmoid_gradient(self.all_preactivations[i-1]))
        elif self.activation=='relu':
          dhidden[self.all_activations[i-1]<=0] = 0
        elif self.activation=='tanh':
          dhidden = np.multiply(dhidden, self._tanh_gradient(self.all_preactivations[i-1]))
        elif self.activation=='linear':
          dhidden = dhidden
      else:
        self.all_dWs[i] = np.dot(input_data.T,dhidden)
        self.all_dBs[i] = np.sum(dhidden, axis=0)

  def fit(self, input_data, input_labels, batch_size = None, epochs = 10):
    num_samples = input_data.shape[0]
    if batch_size is None:
      batch_size = num_samples

    split_indices = np.arange(0, num_samples, batch_size)[1:]

    data_batches = np.array_split(input_data, split_indices, axis=0)
    labels_batches = np.array_split(input_labels, split_indices, axis=0)
    num_batches = len(labels_batches)

    for i in range(epochs):
      for j in range(num_batches):
        each_batch_data = data_batches[j]
        each_batch_labels = labels_batches[j]

        softmax_output = self._forward(each_batch_data)
        loss_value = self._cross_entropy_loss(softmax_output, each_batch_labels)

        if (j+1)%100 == 0 or j==0:
          print('epoch {0}, batch: {1}, loss: {2}'.format(i+1,j+1,loss_value))

        self._backward(softmax_output, each_batch_data, each_batch_labels)

        for k in range(len(self.all_dWs)):
          self.all_weights[k] += -self.learning_rate*self.all_dWs[k]
          self.all_biases[k] += -self.learning_rate*self.all_dBs[k]
    
  def predict(self, input_data):
    softmax_output = self._forward(input_data)
    return softmax_output

  def score(self, input_data, input_labels):
    softmax_output = self.predict(input_data)
    predicted_labels = np.argmax(softmax_output, axis=1)
    return self._accuracy(predicted_labels, input_labels)

if __name__ == "__main__":
  X_train = load_model('train_data.pickle')
  X_test = load_model('test_data.pickle')

  train_data = X_train['data']
  train_labels = X_train['labels'] 

  test_data = X_test['data']
  test_labels = X_test['labels'] 

  mean = np.mean(train_data)
  std = np.std(train_data)

  train_data = (train_data - mean) / std
  test_data = (test_data - mean) / std

  num_features = train_data.shape[1]
  num_labels = len(np.unique(train_labels))

  clf = NeuralNetwork(6, [num_features, 256, 128, 64, 128, num_labels], 'linear', 0.1)
  clf.fit(train_data, train_labels, batch_size=200, epochs=100)
  print(clf.score(train_data, train_labels))
  print(clf.score(test_data, test_labels))
