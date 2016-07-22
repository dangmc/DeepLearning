import cPickle as pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.dtypes import float32

def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation, :]
  shuffled_labels = labels[permutation, :]
  return shuffled_dataset, shuffled_labels

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)

image_size = 28
num_labels = 10
number_channels = 1



def	Reshape(datasets, labels):
  datasets = datasets.reshape((-1, image_size, image_size, number_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
  return datasets, labels

train_dataset, train_labels = Reshape(train_dataset, train_labels)
valid_dataset, valid_labels = Reshape(valid_dataset, valid_labels)
test_dataset, test_labels = Reshape(test_dataset, test_labels)

print (train_dataset.shape, train_labels.shape)
print (valid_dataset.shape, valid_labels.shape)
print (test_dataset.shape, test_labels.shape)


batch_size = 16
patch_size = 5
depth = 16
number_hidden = 64


graph = tf.Graph()
with graph.as_default():
  def   weight_variable(shape):
    init_weight = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(init_weight)

  def   bias_variable(shape):
    init_bias = tf.zeros(shape=shape)
    return tf.Variable(init_bias)

  """ Data """
  tf_train_datasets = tf.placeholder(float32, shape=(batch_size, image_size, image_size, number_channels))
  tf_train_labels = tf.placeholder(float32, shape=(batch_size, num_labels))
  tf_valid_datasets = tf.constant(valid_dataset)
  tf_test_datasets = tf.constant(test_dataset)

  """ Variable """
  weight_layer_1 = weight_variable(shape=[patch_size, patch_size, number_channels, depth])
  bias_layer_1 = bias_variable(shape=[depth])

  weight_layer_2 = weight_variable(shape=[patch_size, patch_size, depth, depth])
  bias_layer_2 = tf.Variable(tf.constant(1.0, shape=[depth]))

  weight_layer_3 = weight_variable(shape=[image_size // 4 * image_size // 4 * depth, number_hidden])
  bias_layer_3 = tf.Variable(tf.constant(1.0, shape=[number_hidden]))

  weight_layer_4 = weight_variable(shape=[number_hidden, num_labels])
  bias_layer_4 = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  """ Model """

  def model(data):
      conv_net = tf.nn.conv2d(data, weight_layer_1, [1, 2, 2, 1], padding="SAME")
      hidden = tf.nn.relu(conv_net + bias_layer_1)

      conv_net = tf.nn.conv2d(hidden, weight_layer_2, [1, 2, 2, 1], padding="SAME")
      hidden = tf.nn.relu(conv_net + bias_layer_2)

      shape_hidden = hidden.get_shape().as_list()
      reshape = tf.reshape(hidden, [shape_hidden[0], shape_hidden[1] * shape_hidden[2] * shape_hidden[3]])
      hidden = tf.nn.relu(tf.matmul(reshape, weight_layer_3) + bias_layer_3)

      return tf.matmul(hidden, weight_layer_4) + bias_layer_4

  """ Training"""
  logits = model(tf_train_datasets)
  cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(cost_function)

  """ Prediction"""
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_datasets))
  test_prediction = tf.nn.softmax(model(tf_test_datasets))

def   accuracy(predictions, labels):
  return 100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

num_steps = 1001

with tf.Session(graph=graph) as session:

  tf.initialize_all_variables().run()
  print ("Initialized")
  for step in xrange(num_steps):

    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset : (offset + batch_size), :, :, :]
    batch_labels = train_labels[offset : (offset + batch_size), :]

    feed_dict = {tf_train_datasets : batch_data, tf_train_labels : batch_labels}

    _, costFunction, predictions = session.run([optimizer, cost_function, train_prediction], feed_dict=feed_dict)
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, costFunction))
      print('Training accuracy: %.2f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.2f%%' % accuracy(valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.2f%%" % accuracy(test_prediction.eval(), test_labels))
