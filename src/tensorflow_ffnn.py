import tensorflow as tf
import itertools


def multilayer_perceptron(x, layer_dims, scale=None):
    ''' o
       oo
      /ooo\
    x -ooo- y
      \ooo/
       oo
        o
   
    Fully connected / dense network mapping x in R^n to y in R^m with
    layer_dims = [n, ...m]
    '''
    assert x.shape[1] == layer_dims[0]

    if scale is None:
        ndofs = 0
    else:
        if isinstance(scale, tf.Variable):
            ndofs = 1
            scale = itertools.repeat(scale)
        else:
            ndofs = len(scale)
            if ndofs == 1:
                scale = itertools.repeat(scale[0])
            else:
                assert ndofs == len(layer_dims)-1
                scale = iter(scale)

    layer_i = x  # The previous one
    # Buld graph for all up to last hidden
    for dim_i, dim_o in zip(layer_dims[:-2], layer_dims[1:]):
        # Random weights
        weights = tf.Variable(tf.truncated_normal(shape=[dim_i, dim_o], stddev=0.1))
        # NOTE: for fitting it seems better to have bias as constant
        biases = tf.Variable(tf.constant(0.1, shape=[dim_o]))

        ndofs += np.prod(weights.shape) + np.prod(biases.shape)
        
        layer_o = tf.add(tf.matmul(layer_i, weights), biases)
        # With ReLU activation
        layer_o = tf.nn.relu(layer_o)

        # Common scaling for ReLU in each layer. FIXME: clipping
        if scale is not None:
            layer_o = next(scale)*tf.nn.relu(layer_o)

        layer_i = layer_o

    # Now from hidden to output
    dim_i, dim_o = layer_dims[-2:]
    weights = tf.Variable(tf.truncated_normal([dim_i, dim_o], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[dim_o]))
    ndofs += np.prod(weights.shape) + np.prod(biases.shape)
    
    layer_o = tf.add(tf.matmul(layer_i, weights), biases)

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

    def predict(sess, NN, x, x_values):
        y_ = [sess.run(NN, feed_dict={x: np.array([[xi]])}) for xi in x_values]
        return np.array(y_).flatten()

    # Spec the architecture
    layer_dims = [1, 2, 4, 2, 1]

    x = tf.placeholder(tf.float32, [None, layer_dims[0]])
    y = tf.placeholder(tf.float32, [None, layer_dims[-1]])
    # Activation function scaling
    get_scalar = lambda v: tf.Variable(tf.constant(v), constraint=lambda t: tf.clip_by_value(t, 1E-8, 100))
    
    scale = [get_scalar(1+0.1*i) for i in range(len(layer_dims)-1)]
    # scale = None
    
    # The net is now
    NN, ndofs = multilayer_perceptron(x, layer_dims, scale=scale)

    # The loss functional
    loss = tf.reduce_mean(tf.square(NN - y))  # reduce_[sum, mean]
    
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

    unit_interval = np.linspace(0, 1, 1000)
    plt.figure()

    if scale is not None:
        scale_values = [list() for _ in range(len(scale))]
    
    for step in range(training_epochs):
        x_data = np.random.rand(batch_size, 1).astype(np.float32)
        y_data = target_foo(x_data)
        sess.run(train, feed_dict={x: x_data, y: y_data})
    
        if step % display_step == 0:
            x_test = np.random.rand(1, 1).astype(np.float32)
            y_test = target_foo(x_test)

            error = sess.run(loss, feed_dict={x: x_test, y: y_test})
            if scale is not None:
                for scale_value, scale_ in zip(scale_values, scale):
                    scale_value.append(sess.run(scale_))
        
            print('At step %d error %g' % (step, error))

            (step % (4*display_step) == 0) and plt.plot(unit_interval, predict(sess, NN, x, unit_interval), label=str(step))
            
    print('Network size', ndofs)
    
    plt.plot(unit_interval, target_foo(unit_interval), label='truth')
    plt.legend()

    if scale is not None:
        plt.figure()
        for idx, values in enumerate(scale_values):
            plt.plot(values, label=str(idx))
        plt.legend()
    plt.show()
