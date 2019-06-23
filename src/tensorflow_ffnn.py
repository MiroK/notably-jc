import tensorflow as tf

def multilayer_perceptron(x, layer_dims, scale=None):
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
    
    ndofs = 0
    layer_i = x  # The previous one
    # Buld graph for all up to last hidden
    for dim_i, dim_o in zip(layer_dims[:-2], layer_dims[1:]):
        weights = tf.Variable(tf.truncated_normal([dim_i, dim_o], stddev=0.1))
        biases = tf.Variable(tf.constant(0.1, shape=[dim_o]))

        ndofs += np.prod(weights.shape) + np.prod(biases.shape) + 1
        
        layer_o = tf.add(tf.matmul(layer_i, weights), biases)
        # With ReLU activation
        layer_o = tf.nn.relu(layer_o)

        # Common scaling for ReLU in each layer. FIXME: clipping
        if scale is not None: layer_o = scale*tf.nn.relu(layer_o)

        layer_i = layer_o

    # Now from hidden to output
    dim_i, dim_o = layer_dims[-2:]
    weights = tf.Variable(tf.truncated_normal([dim_i, dim_o], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[dim_o]))
    ndofs += np.prod(weights.shape) + np.prod(biases.shape)
    
    layer_o = tf.add(tf.matmul(layer_i, weights), biases)

    if scale is not None: ndofs += 1

    return layer_o, ndofs


# --------------------------------------------------------------------


if __name__ == '__main__':
    import matplotlib.pyplot as plt
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
    layer_dims = [1, 2, 4, 2, 1]

    x = tf.placeholder(tf.float32, [None, layer_dims[0]])
    y = tf.placeholder(tf.float32, [None, layer_dims[-1]])
    # Activation function scaling
    scale = tf.Variable(tf.constant(1.), constraint=lambda t: tf.clip_by_value(t, 1E-6, 100))

    # The net is now
    NN, ndofs = multilayer_perceptron(x, layer_dims, scale=scale)

    # The loss functional
    loss = tf.reduce_mean(tf.square(NN - y))
    
    learning_rate = 1E-4
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    training_epochs = 20000
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
            x_test = np.random.rand(1, 1).astype(np.float32)
            y_test = target_foo(x_test)

            error = sess.run(loss, feed_dict={x: x_test, y: y_test})
            scale_ = sess.run(scale)
            print('At step %d error %g, scale is %g' % (step, error, scale_))

    x_ = np.linspace(0, 1, 1000)
    y_ = [sess.run(NN, feed_dict={x: np.array([[xi]])}) for xi in x_]
    y_ = np.array(y_).flatten()

    print('Network size', ndofs)
    
    plt.figure()
    plt.plot(x_, y_, label='num')
    plt.plot(x_, target_foo(x_), label='truth')
    plt.legend()
    plt.show()
