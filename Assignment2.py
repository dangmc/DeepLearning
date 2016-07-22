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

def	Reshape(datasets, labels):
  datasets = datasets.reshape((-1, image_size * image_size)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
  return datasets, labels

train_dataset, train_labels = Reshape(train_dataset, train_labels)
valid_dataset, valid_labels = Reshape(valid_dataset, valid_labels)
test_dataset, test_labels = Reshape(test_dataset, test_labels)

print (train_dataset.shape, train_labels.shape)
print (valid_dataset.shape, valid_labels.shape)
print (test_dataset.shape, test_labels.shape)


batch_size = 128
num_hidden_neuron = 1024

graph = tf.Graph()
with graph.as_default():
  def   weight_variable(shape):
    init_weight = tf.truncated_normal(shape=shape)
    return tf.Variable(init_weight)

  def   bias_variable(shape):
    init_bias = tf.zeros(shape=shape)
    return tf.Variable(init_bias)

  tf_train_datasets = tf.placeholder(float32, shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(float32, shape=(batch_size, num_labels))

  tf_valid_datasets = tf.constant(valid_dataset)
  tf_test_datasets = tf.constant(test_dataset)

  keep_prob = tf.placeholder(float32)

  weight_1 = weight_variable(shape=[image_size * image_size, num_hidden_neuron])
  weight_2 = weight_variable(shape=[num_hidden_neuron, num_labels ])

  bias_1 = bias_variable(shape=[num_hidden_neuron])
  bias_2 = bias_variable(shape=[num_labels])

  activation_hidden = tf.nn.relu(tf.matmul(tf_train_datasets, weight_1) + bias_1)
  activation_hidden_dropout = tf.nn.dropout(activation_hidden, keep_prob=keep_prob)
  logits = tf.matmul(activation_hidden_dropout, weight_2) + bias_2

  weight_decay = tf.constant(5.0) * (tf.nn.l2_loss(weight_1) + tf.nn.l2_loss(weight_2))/ (train_labels.shape[0])

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + weight_decay
  optimizer = tf.train.GradientDescentOptimizer(3.0).minimize(loss)


  train_prediction = tf.nn.softmax(logits)

  valid_activation = tf.nn.relu(tf.matmul(tf_valid_datasets, weight_1) + bias_1)
  valid_prediction = tf.nn.softmax(tf.matmul(valid_activation, weight_2) + bias_2)

  test_activation = tf.nn.relu(tf.matmul(tf_test_datasets, weight_1) + bias_1)
  test_prediction = tf.nn.softmax(tf.matmul(test_activation, weight_2) + bias_2)


def   accuracy(predictions, labels):
  return 100 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

num_steps = 10001

def   weight_random(shape):
  init_weight = tf.truncated_normal(shape=shape)
  return tf.Variable(init_weight)

def   bias_random(shape):
  init_bias = tf.truncated_normal(shape=shape)
  return tf.Variable(init_bias)

with tf.Session(graph=graph) as session:


  tf.initialize_all_variables().run()
  print ("Initialized")
  for step in xrange(num_steps):

    if (step % 2000 == 0):
      train_dataset, train_labels = randomize(train_dataset, train_labels)

    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

    batch_data = train_dataset[offset : (offset + batch_size), :]
    batch_labels = train_labels[offset : (offset + batch_size), :]

    feed_dict = {tf_train_datasets : batch_data, tf_train_labels : batch_labels, keep_prob : 0.5}

    _, costFunction, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 1000 == 0):
      print('Loss at step %d: %f' % (step, costFunction))
      print('Training accuracy: %.2f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.2f%%' % accuracy(valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.2f%%" % accuracy(test_prediction.eval(), test_labels))
