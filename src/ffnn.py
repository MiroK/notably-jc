import numpy as np


def relu(y): return np.maximum(0, y)


def identity(x): return x


def setup_FFNN(layers):
    '''
    Fully connected feed forward neural network with the architecture 
    speficied by sizes of hidden layers

         -M
        NxMxL
       /NxMxL \  
      / NxMxL  \
    i --NxMxL  -o
      \ NxMxL  /
       \NxMxL /
        NxMxL  
         -M
    '''
    assert len(layers) >= 1
    # One node input, then hidden layers, then one output
    sizes = np.r_[1, layers, 1]
    # Nonlinearities for transitions between layers. To get to output
    # use identity 
    nlins = [relu]*len(layers) + [identity]

    def NN(x, coefs, sizes=sizes, nlins=nlins):
        # Operations to get from l-1 to l
        index = 0
        for l, nlin in enumerate(nlins, 1):
            size_out, size_in = sizes[l], sizes[l-1]
            # Weights are first in the coefs verctor; building matrix
            # to go from input to output space
            weights = coefs[index:(index+size_out*size_in)]
            W = weights.reshape((size_out, size_in))
            
            index += size_out*size_in
            # Then biases; one for each output
            biases = coefs[index:index+size_out].reshape((-1, 1))
            index += size_out

            x = nlin(W.dot(x) - biases)

        # print '>>>', x[0][0], output
        return x[0][0]
    # Return the network and the coefs size
    dofs = sum(i*o + o for i, o in zip(sizes[:-1], sizes[1:]))
    
    return NN, dofs


# --------------------------------------------------------------------


if __name__ == '__main__':
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt
    
    # Want to fit 
    def f(x): 
        val = 0 
        if x >= 0.25 and x <= 0.5: val = 4*x - 1  
        if x >= 0.5 and x <= 0.75: val = -4*x + 3   
        #  if x >= 0.5: val = x - 0.5  
        #  if x <= 0.5: val = 0.5 - x   
        return val

    # On interval
    dx = 0.01
    mesh = np.arange(0, 1+dx/2, dx)

    # Using NN as
    NN, dofs = setup_FFNN(layers=np.array([20]))
    # Init
    coefs = np.random.rand(dofs)

    #NN(0.2, coefs)

    #exit()
    truth = map(f, mesh)
    history = []
    # The error on interval given current coefs is
    def goal(coefs, NN=NN, mesh=mesh, truth=truth, history=history):
        predicted = np.array([NN(x, coefs) for x in mesh])
        error = np.linalg.norm(predicted - truth, 2)/len(truth)
        history.append(error)
        
        print error, np.linalg.norm(coefs)

        return error

    ret = minimize(goal, coefs, method='BFGS', options={'disp': True})
    coefs = ret.x
    print np.linalg.norm(coefs)
    print coefs
    
    predicted = [NN(x, coefs) for x in mesh]
    
    plt.figure()
    plt.plot(mesh, truth)
    plt.plot(mesh, predicted)

    plt.figure()
    plt.semilogy(np.arange(1, len(history)+1), history)
    
    plt.show()
