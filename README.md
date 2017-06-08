# Deep Learning
## Project: Image Classification
### Files Description
It contains four files:
- `project description.md`: Project overview, highlights, evaluation and software requirement. **Read Firstly**
- `README.md`: this file.
- `image_classification.ipynb`: This is the main file where you will answer questions and provide an analysis for your work.
- `problem_unittests.py`: This Python script provides supplementary test units.
- `helper.py`: This Python script provides data preprocess. 

### Run
In a command window (OS: Win7), navigate to the top-level project directory that contains this README and run one of the following commands:
`jupyter notebook image_classification.ipynb`

## Project Implementation
### Preprocessing
#### Normalization
The `normalize` function normalizes image data in the range of 0 to 1, inclusive.
```
def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    x =x*1.0/255
    x = np.array(x)
    return x
```
#### Output Label
The `one_hot_encode` function encodes labels to one-hot encodings.
```
def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    one_hot_labels = np.zeros((len(x),10))
    for idx, val in enumerate(x):
        one_hot_labels[idx,val] = 1
        
    return one_hot_labels
```
### Neural Network Layers
#### Data Transformation
The neural net inputs functions have all returned the correct TF Placeholder.

- Implement `neural_net_image_input`
    - Return a [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder)
    - Set the shape using `image_shape` with batch size set to `None`.
    - Name the TensorFlow placeholder "x" using the TensorFlow name parameter in the TF Placeholder.
- Implement `neural_net_label_input`
    - Return a TF Placeholder
    - Set the shape using `n_classes` with batch size set to `None`.
    - Name the TensorFlow placeholder "y" using the TensorFlow name parameter in the TF Placeholder.
- Implement `neural_net_keep_prob_input`
    - Return a TF Placeholder for dropout keep probability.
    - Name the TensorFlow placeholder "keep_prob" using the TensorFlow `name` parameter in the TF Placeholder.

```
import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    
    x = tf.placeholder(tf.float32, shape=[None,image_shape[0],image_shape[1],image_shape[2]], name="x")
    return x


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    y = tf.placeholder(tf.float32, shape=[None,n_classes], name="y")
    return y


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return keep_prob
```
#### Convolution and Max Pooling Layer
The `conv2d_maxpool` function applies convolution and max pooling to a layer.

The convolutional layer should use a nonlinear activation.

This function shouldn’t use any of the tensorflow functions in the `tf.contrib` or `tf.layers` namespace.

- Create the weight and bias using `conv_ksize`, `conv_num_outputs` and the shape of `x_tensor`.
- Apply a convolution to `x_tensor` using weight and `conv_strides`.
    - We recommend you use same padding, but you're welcome to use any padding.
- Add bias
- Add a nonlinear activation to the convolution.
- Apply Max Pooling using `pool_ksize` and `pool_strides`.
    - We recommend you use same padding, but you're welcome to use any padding.

```
def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function
    #print(x_tensor.shape[1]) # 32
    #print(conv_ksize) # (2, 2)
    #print(conv_num_outputs) # 10
    #print(type(conv_num_outputs))
    #print((x_tensor))
    #print(type(x_tensor.shape[3]))
    #print(x_tensor.shape) # (?, 32, 32, 5)

    #print(type(x_tensor.shape)) # <class 'tensorflow.python.framework.tensor_shape.TensorShape'>
    #print(type(tf.to_int32(x_tensor.shape[3])))
    #print(type(conv_num_outputs))
    #print(type(x_tensor)) # <class 'tensorflow.python.framework.ops.Tensor'>    
    #print(type(conv_strides[0]))
        
    weight = tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1],tf.to_int32(x_tensor.shape[3]),conv_num_outputs])) # (height, width, input_depth, output_depth)
    #print(type(weight)) # <class 'tensorflow.python.ops.variables.Variable'>
    bias = tf.Variable(tf.zeros([conv_num_outputs]))

    conv_layer = tf.nn.conv2d(x_tensor,weight, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    # Add bias
    conv_layer = tf.nn.bias_add(conv_layer, bias)
    # Apply activation function
    conv_layer = tf.nn.relu(conv_layer)                
    # Apply Max Pooling
    conv_layer = tf.nn.max_pool(
        conv_layer,
        ksize=[1, pool_ksize[0], pool_ksize[1], 1],
        strides=[1, pool_strides[0], pool_strides[1], 1],
        padding='SAME')

    return conv_layer 
```
#### Flatten Layer
The `flatten` function flattens a tensor without affecting the batch size.
Implement the `flatten` function to change the dimension of `x_tensor` from a 4-D tensor to a 2-D tensor. The output should be the shape (Batch Size, Flattened Image Size).
```
def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] 
    return tf.contrib.layers.flatten(x_tensor)
```
#### Fully-Connected Layer
The `fully_conn` function creates a fully connected layer with a nonlinear activation.
Implement the `fully_conn` function to apply a fully connected layer to `x_tensor` with the shape (Batch Size, num_outputs). 
```
def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    return tf.contrib.layers.fully_connected(x_tensor, num_outputs,tf.nn.relu)
```
#### Output Layer
The `output` function creates an output layer with a linear activation.
Implement the `output` function to apply a fully connected layer to `x_tensor` with the shape (Batch Size, num_outputs). 
```
def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    return tf.contrib.layers.fully_connected(x_tensor, num_outputs,None)
```
### Neural Network Architecture
#### Create Convolutional Model
The `conv_net` function creates a convolutional model and returns the logits. Dropout should be applied to alt least one layer.

Implement the function `conv_net` to create a convolutional neural network model. The function takes in a batch of images, `x`, and outputs logits. Use the layers you created above to create this model:
- Apply 1, 2, or 3 Convolution and Max Pool layers
- Apply a Flatten Layer
- Apply 1, 2, or 3 Fully Connected Layers
- Apply an Output Layer
- Return the output
- Apply [TensorFlow's Dropout](https://www.tensorflow.org/api_docs/python/tf/nn/dropout) to one or more layers in the model using keep_prob.

```
def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)
    x = conv2d_maxpool(x, 90, (3,3), (1,1), (2,2), (2,2))

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    x = flatten(x)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)
    x= fully_conn(x, 500)
    x=tf.nn.dropout(x, keep_prob)
    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)
    x=output(x,10)
    
    # TODO: return output
    return x
```
### Neural Network Training
#### Single Optimization
The `train_neural_network` function optimizes the neural network.
Implement the function `train_neural_network` to do a single optimization. The optimization should use `optimizer` to optimize in session with a `feed_dict` of the following:
- `x` for image input
- `y` for labels
- `keep_prob` for keep probability for dropout
This function will be called for each batch, so `tf.global_variables_initializer()` has already been called.
Note: Nothing needs to be returned. This function is only optimizing the neural network.
```
def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    session.run(optimizer, feed_dict={
        x: feature_batch,
        y: label_batch,
        keep_prob: keep_probability})
    pass

```
#### Show Stats
The `print_stats` function prints loss and validation accuracy.
Implement the function `print_stats` to print loss and validation accuracy. Use the global variables `valid_features` and `valid_labels` to calculate validation accuracy. Use a keep probability of `1.0` to calculate the loss and validation accuracy.
```
def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    # Calculate batch loss and accuracy
    loss = session.run(cost, feed_dict={
        x: feature_batch,
        y: label_batch,
        keep_prob: 1.})
    validation_accuracy  = session.run(accuracy, feed_dict={
        x: valid_features,
        y: valid_labels,
        keep_prob: 1.})
    print('loss: {}, validation_accuracy: {}'.format(loss, validation_accuracy))
    pass
```
#### Hyperparameters
The hyperparameters have been set to reasonable numbers.
Tune the following parameters:
- Set `epochs` to the number of iterations until the network stops learning or start overfitting
- Set `batch_size` to the highest number that your machine has memory for. Most people set them to common sizes of memory:
    - 64
    - 128
    - 256
    - ...
- Set `keep_probability` to the probability of keeping a node using dropout
```
# TODO: Tune Parameters
epochs = 200
batch_size = 256
keep_probability = 0.5
```
#### Train on a Single CIFAR-10 Batch
The neural network validation and test accuracy are similar. Their accuracies are greater than 50%.




