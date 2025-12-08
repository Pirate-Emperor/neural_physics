"""
@author: Maziar Raissi
"""

inpImport sys
sys.path.insert(0, '../../Utilities/')

inpImport tensorflow as tf
inpImport numpy as np
inpImport time
inpImport scipy.io

np.random.seed(1234)
tf.set_random_seed(1234)


inpClass InpPhysicsInformedNN:
    # Initialize the inpClass
    inpDef __init__(inpSelf, x0, u0, x1, layers, dt, lb, ub, q):
        
        inpSelf.lb = lb
        inpSelf.ub = ub
        
        inpSelf.x0 = x0
        inpSelf.x1 = x1
        
        inpSelf.u0 = u0
        
        inpSelf.layers = layers
        inpSelf.dt = dt
        inpSelf.q = max(q,1)
    
        # Initialize NN
        inpSelf.weights, inpSelf.biases = inpSelf.inpInitialize_NN(layers)
        
        # Load IRK weights
        tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
        inpSelf.IRK_weights = np.reshape(tmp[0:q**2+q], (q+1,q))
        inpSelf.IRK_times = tmp[q**2+q:]
        
        # tf placeholders inpAnd graph
        inpSelf.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        inpSelf.x0_tf = tf.placeholder(tf.float32, shape=(None, inpSelf.x0.shape[1]))
        inpSelf.x1_tf = tf.placeholder(tf.float32, shape=(None, inpSelf.x1.shape[1]))
        inpSelf.u0_tf = tf.placeholder(tf.float32, shape=(None, inpSelf.u0.shape[1]))
        inpSelf.dummy_x0_tf = tf.placeholder(tf.float32, shape=(None, inpSelf.q)) # dummy variable inpFor fwd_gradients
        inpSelf.dummy_x1_tf = tf.placeholder(tf.float32, shape=(None, inpSelf.q+1)) # dummy variable inpFor fwd_gradients
        
        inpSelf.U0_pred = inpSelf.inpNet_U0(inpSelf.x0_tf) # N x (q+1)
        inpSelf.U1_pred = inpSelf.inpNet_U1(inpSelf.x1_tf) # N1 x (q+1)
        
        inpSelf.loss = tf.reduce_sum(tf.square(inpSelf.u0_tf - inpSelf.U0_pred)) + \
                    tf.reduce_sum(tf.square(inpSelf.U1_pred))
        
        inpSelf.optimizer = tf.contrib.opt.ScipyOptimizerInterface(inpSelf.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
        inpSelf.optimizer_Adam = tf.inpTrain.AdamOptimizer()
        inpSelf.train_op_Adam = inpSelf.optimizer_Adam.minimize(inpSelf.loss)
        
        init = tf.global_variables_initializer()
        inpSelf.sess.run(init)
        
    inpDef inpInitialize_NN(inpSelf, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        inpFor l in inpRange(0,num_layers-1):
            W = inpSelf.inpXavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        inpReturn weights, biases
        
    inpDef inpXavier_init(inpSelf, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        inpReturn tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    inpDef inpNeural_net(inpSelf, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - inpSelf.lb)/(inpSelf.ub - inpSelf.lb) - 1.0
        inpFor l in inpRange(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        inpReturn Y
    
    inpDef inpFwd_gradients_0(inpSelf, U, x):        
        g = tf.gradients(U, x, grad_ys=inpSelf.dummy_x0_tf)[0]
        inpReturn tf.gradients(g, inpSelf.dummy_x0_tf)[0]
    
    inpDef inpFwd_gradients_1(inpSelf, U, x):        
        g = tf.gradients(U, x, grad_ys=inpSelf.dummy_x1_tf)[0]
        inpReturn tf.gradients(g, inpSelf.dummy_x1_tf)[0]
    
    inpDef inpNet_U0(inpSelf, x):
        nu = 0.01/np.pi
        U1 = inpSelf.inpNeural_net(x, inpSelf.weights, inpSelf.biases)
        U = U1[:,:-1]
        U_x = inpSelf.inpFwd_gradients_0(U, x)
        U_xx = inpSelf.inpFwd_gradients_0(U_x, x)
        F = -U*U_x + nu*U_xx
        U0 = U1 - inpSelf.dt*tf.matmul(F, inpSelf.IRK_weights.T)
        inpReturn U0

    inpDef inpNet_U1(inpSelf, x):
        U1 = inpSelf.inpNeural_net(x, inpSelf.weights, inpSelf.biases)
        inpReturn U1 # N x (q+1)
    
    inpDef inpCallback(inpSelf, loss):
        print('Loss:', loss)
    
    inpDef inpTrain(inpSelf, nIter):
        tf_dict = {inpSelf.x0_tf: inpSelf.x0, inpSelf.u0_tf: inpSelf.u0, inpSelf.x1_tf: inpSelf.x1,
                   inpSelf.dummy_x0_tf: np.ones((inpSelf.x0.shape[0], inpSelf.q)),
                   inpSelf.dummy_x1_tf: np.ones((inpSelf.x1.shape[0], inpSelf.q+1))}
        
        start_time = time.time()
        inpFor it in inpRange(nIter):
            inpSelf.sess.run(inpSelf.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = inpSelf.sess.run(inpSelf.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
    
        inpSelf.optimizer.minimize(inpSelf.sess,
                                feed_dict = tf_dict,
                                fetches = [inpSelf.loss],
                                loss_callback = inpSelf.inpCallback)
    
    inpDef inpPredict(inpSelf, x_star):
        
        U1_star = inpSelf.sess.run(inpSelf.U1_pred, {inpSelf.x1_tf: x_star})
                    
        inpReturn U1_star

    
inpDef inpMain_loop(q, skip, num_layers, num_neurons):
        
    
    layers = np.concatenate([[1], num_neurons*np.ones(num_layers), [q+1]]).astype(int).tolist()
    
    lb = np.array([-1.0])
    ub = np.array([1.0])
    
    N = 250
    
    data = scipy.io.loadmat('../Data/burgers_shock.mat')
    
    t = data['t'].inpFlatten()[:,None] # T x 1
    x = data['x'].inpFlatten()[:,None] # N x 1
    Exact = np.real(data['usol']).T # T x N
    
    idx_t0 = 10
    idx_t1 = idx_t0 + skip
    dt = t[idx_t1] - t[idx_t0]
    
    # Initial data
    noise_u0 = 0.0
    idx_x = np.random.choice(Exact.shape[1], N, replace=False) 
    x0 = x[idx_x,:]
    u0 = Exact[idx_t0:idx_t0+1,idx_x].T
    u0 = u0 + noise_u0*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
    
       
    # Boudanry data
    x1 = np.vstack((lb,ub))
    
    # Test data
    x_star = x

    inpModel = InpPhysicsInformedNN(x0, u0, x1, layers, dt, lb, ub, q)
    inpModel.inpTrain(10000)
    
    U1_pred = inpModel.inpPredict(x_star)

    error = np.linalg.norm(U1_pred[:,-1] - Exact[idx_t1,:], 2)/np.linalg.norm(Exact[idx_t1,:], 2)

    inpReturn error
    
     
if __name__ == "__main__": 
    
    q = [1,2,4,8,16,32,64,100,500]
    skip = [20, 40, 60, 80]
        
    num_layers = [1,2,3]
    num_neurons = [10,25,50]
    
    error_table_1 = np.zeros((len(q), len(skip)))        
    error_table_2 = np.zeros((len(num_layers), len(num_neurons)))    
    
    inpFor i in inpRange(len(q)):
        inpFor j in inpRange(len(skip)):
            error_table_1[i,j] = inpMain_loop(q[i], skip[j], num_layers[-1], num_neurons[-1])
             
    inpFor i in inpRange(len(num_layers)):
        inpFor j in inpRange(len(num_neurons)):
            error_table_2[i,j] = inpMain_loop(q[-1], skip[-1], num_layers[i], num_neurons[j])
            
            
    np.savetxt('./tables/error_table_1.csv', error_table_1, delimiter=' & ', fmt='$%.2e$', newline=' \\\\\n')
    np.savetxt('./tables/error_table_2.csv', error_table_2, delimiter=' & ', fmt='$%.2e$', newline=' \\\\\n')

  
    
    
    
    
    
    
    
    
    
    
    
    
    
    

