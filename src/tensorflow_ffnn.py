import tensorflow as tf


def multilayer_perceptron(x, layer_dims):
    ''' o
       oo
      /ooo\
    x -ooo- y
      \ooo/
       oo
        o
   
    Fully connected
    '''
    assert x.shape[1] == layer_dims[0]

    layer_i = x  # The previous one
    # Buld graph for all up to last hidden
    for dim_i, dim_o in zip(layer_dims[:-2], layer_dims[1:]):

        weights = tf.Variable(tf.truncated_normal([dim_o, dim_i], stddev=0.1))
        print (dim_i, dim_o), weights.shape, layer_i.shape
        biases = tf.Variable(tf.constant(0.1, shape=[dim_o]))
        
        layer_o = tf.add(tf.matmul(weights, layer_i), biases)
        # With ReLU activation
        layer_o = tf.nn.relu(layer_o)

        layer_i = layer_o

    # Now from hidden to output
    dim_i, dim_o = layer_dims[-2:]
    weights = tf.Variable(tf.truncated_normal([dim_o, dim_i], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[dim_o]))
    
    layer_o = tf.add(tf.matmul(weights, layer_i), biases)

    return layer_o


# --------------------------------------------------------------------


if __name__ == '__main__':
    import numpy as np

    
    def target_foo(x):
        '''The hat function'''
        y = np.zeros_like(x)

        idx = np.where(np.logical_and(x >= 0.25, x <= 0.5))[0] 
        y[idx] = 4*x[idx] - 1
    
        idx = np.where(np.logical_and(x >= 0.5, x <= 0.75))[0]
        y[idx] = -4*x[idx] + 3

        return y

    # Spec the architecture
    layer_dims = [1, 50, 10, 1]

    x = tf.placeholder(tf.float32, [None, layer_dims[0]])
    y = tf.placeholder(tf.float32, [None, layer_dims[1]])

    # The net is now
    NN = multilayer_perceptron(x, layer_dims)

    # The loss functional
    loss = tf.reduce_mean(tf.square(NN - y))
    
    learning_rate = 1E-4
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    training_epochs = 10000
    batch_size = 500
    display_step = 500

    # Before starting, initialize the variables
    init = tf.initialize_all_variables()

    # Launch the graph.
    sess = tf.Session()
    sess.run(init)

    for step in range(training_epochs):
        x_data = np.random.rand(batch_size, 1).astype(np.float32)
        y_data = target_foo(x_data)
        sess.run(train, feed_dict={x: x_data, y: y_data})
    
        if step % display_step == 0:
            curX = np.random.rand(1, 1).astype(np.float32)
            curY = target_foo(curX)

            curPrediction = sess.run(NN, feed_dict={x_data: curX})
            curLoss = sess.run(loss, feed_dict={x_data: curX, y_data: curY})
        
            print 'At step %d error %g' % (step, curLoss)

# #x = np.linspace(0, 1, 1000)
# #y = [sess.run(NN, feed_dict={x_data: np.array([[xi]])}) for xi in x]
# #y = np.array(y).flatten()

# #import matplotlib.pyplot as plt

# #plt.figure()
# #plt.plot(x, y)
# #plt.show()
