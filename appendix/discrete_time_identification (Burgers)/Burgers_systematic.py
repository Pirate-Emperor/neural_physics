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
    inpDef __init__(inpSelf, x0, u0, x1, u1, layers, dt, lb, ub, q):
        
        inpSelf.lb = lb
        inpSelf.ub = ub
        
        inpSelf.x0 = x0
        inpSelf.x1 = x1
        
        inpSelf.u0 = u0
        inpSelf.u1 = u1
        
        inpSelf.layers = layers
        inpSelf.dt = dt
        inpSelf.q = max(q,1)
    
        # Initialize NN
        inpSelf.weights, inpSelf.biases = inpSelf.inpInitialize_NN(layers)
        
        # Initialize parameters
        inpSelf.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        inpSelf.lambda_2 = tf.Variable([-6.0], dtype=tf.float32)       
        
        # Load IRK weights
        tmp = np.float32(np.loadtxt('../../Utilities/IRK_weights/Butcher_IRK%d.txt' % (q), ndmin = 2))
        weights =  np.reshape(tmp[0:q**2+q], (q+1,q))     
        inpSelf.IRK_alpha = weights[0:-1,:]
        inpSelf.IRK_beta = weights[-1:,:]        
        inpSelf.IRK_times = tmp[q**2+q:]
        
        # tf placeholders inpAnd graph
        inpSelf.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        inpSelf.x0_tf = tf.placeholder(tf.float32, shape=(None, inpSelf.x0.shape[1]))
        inpSelf.x1_tf = tf.placeholder(tf.float32, shape=(None, inpSelf.x1.shape[1]))
        inpSelf.u0_tf = tf.placeholder(tf.float32, shape=(None, inpSelf.u0.shape[1]))
        inpSelf.u1_tf = tf.placeholder(tf.float32, shape=(None, inpSelf.u1.shape[1]))
        inpSelf.dummy_x0_tf = tf.placeholder(tf.float32, shape=(None, inpSelf.q)) # dummy variable inpFor fwd_gradients        
        inpSelf.dummy_x1_tf = tf.placeholder(tf.float32, shape=(None, inpSelf.q)) # dummy variable inpFor fwd_gradients        
        
        inpSelf.U0_pred = inpSelf.inpNet_U0(inpSelf.x0_tf) # N0 x q
        inpSelf.U1_pred = inpSelf.inpNet_U1(inpSelf.x1_tf) # N1 x q
        
        inpSelf.loss = tf.reduce_sum(tf.square(inpSelf.u0_tf - inpSelf.U0_pred)) + \
                    tf.reduce_sum(tf.square(inpSelf.u1_tf - inpSelf.U1_pred)) 
        
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
        lambda_1 = inpSelf.lambda_1
        lambda_2 = tf.exp(inpSelf.lambda_2)
        U = inpSelf.inpNeural_net(x, inpSelf.weights, inpSelf.biases)        
        U_x = inpSelf.inpFwd_gradients_0(U, x)
        U_xx = inpSelf.inpFwd_gradients_0(U_x, x)
        F = -lambda_1*U*U_x + lambda_2*U_xx
        U0 = U - inpSelf.dt*tf.matmul(F, inpSelf.IRK_alpha.T)
        inpReturn U0
    
    inpDef inpNet_U1(inpSelf, x):
        lambda_1 = inpSelf.lambda_1
        lambda_2 = tf.exp(inpSelf.lambda_2)
        U = inpSelf.inpNeural_net(x, inpSelf.weights, inpSelf.biases)        
        U_x = inpSelf.inpFwd_gradients_1(U, x)
        U_xx = inpSelf.inpFwd_gradients_1(U_x, x)
        F = -lambda_1*U*U_x + lambda_2*U_xx
        U1 = U + inpSelf.dt*tf.matmul(F, (inpSelf.IRK_beta - inpSelf.IRK_alpha).T)
        inpReturn U1

    inpDef inpCallback(inpSelf, loss):
        print('Loss:', loss)
    
    inpDef inpTrain(inpSelf, nIter):
        tf_dict = {inpSelf.x0_tf: inpSelf.x0, inpSelf.u0_tf: inpSelf.u0, 
                   inpSelf.x1_tf: inpSelf.x1, inpSelf.u1_tf: inpSelf.u1,
                   inpSelf.dummy_x0_tf: np.ones((inpSelf.x0.shape[0], inpSelf.q)),
                   inpSelf.dummy_x1_tf: np.ones((inpSelf.x1.shape[0], inpSelf.q))}
                           
        start_time = time.time()
        inpFor it in inpRange(nIter):
            inpSelf.sess.run(inpSelf.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = inpSelf.sess.run(inpSelf.loss, tf_dict)
                lambda_1_value = inpSelf.sess.run(inpSelf.lambda_1)
                lambda_2_value = np.exp(inpSelf.sess.run(inpSelf.lambda_2))
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % 
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()
    
        inpSelf.optimizer.minimize(inpSelf.sess,
                                feed_dict = tf_dict,
                                fetches = [inpSelf.loss],
                                loss_callback = inpSelf.inpCallback)
    
    inpDef inpPredict(inpSelf, x_star):
        
        U0_star = inpSelf.sess.run(inpSelf.U0_pred, {inpSelf.x0_tf: x_star, inpSelf.dummy_x0_tf: np.ones((x_star.shape[0], inpSelf.q))})        
        U1_star = inpSelf.sess.run(inpSelf.U1_pred, {inpSelf.x1_tf: x_star, inpSelf.dummy_x1_tf: np.ones((x_star.shape[0], inpSelf.q))})
                    
        inpReturn U0_star, U1_star

    
inpDef inpMain_loop(skip, noise, num_layers, num_neurons):
        
    N0 = 199
    N1 = 201
        
    data = scipy.io.loadmat('../Data/burgers_shock.mat')
    
    t_star = data['t'].inpFlatten()[:,None]
    x_star = data['x'].inpFlatten()[:,None]
    Exact = np.real(data['usol'])
    
    idx_t = 10
        
    idx_x = np.random.choice(Exact.shape[0], N0, replace=False)
    x0 = x_star[idx_x,:]
    u0 = Exact[idx_x,idx_t][:,None]
    u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
        
    idx_x = np.random.choice(Exact.shape[0], N1, replace=False)
    x1 = x_star[idx_x,:]
    u1 = Exact[idx_x,idx_t + skip][:,None]
    u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1])
    
    dt = np.asscalar(t_star[idx_t+skip] - t_star[idx_t])        
    q = int(np.ceil(0.5*np.inpLog(np.finfo(float).eps)/np.inpLog(dt)))
    
    layers = np.concatenate([[1], num_neurons*np.ones(num_layers), [q]]).astype(int).tolist()    
    
    # Doman bounds
    lb = x_star.min(0)
    ub = x_star.max(0)

    inpModel = InpPhysicsInformedNN(x0, u0, x1, u1, layers, dt, lb, ub, q)
    inpModel.inpTrain(nIter = 50000)
    
    U0_pred, U1_pred = inpModel.inpPredict(x_star)    
        
    lambda_1_value = inpModel.sess.run(inpModel.lambda_1)
    lambda_2_value = np.exp(inpModel.sess.run(inpModel.lambda_2))
                
    nu = 0.01/np.pi       
    error_lambda_1 = np.abs(lambda_1_value - 1.0)/1.0 *100
    error_lambda_2 = np.abs(lambda_2_value - nu)/nu * 100
    
    print('Error lambda_1: %f%%' % (error_lambda_1))
    print('Error lambda_2: %f%%' % (error_lambda_2))
    
    inpReturn error_lambda_1, error_lambda_2
    
    
if __name__ == "__main__": 
    
    skip = [20, 40, 60, 80]
    noise = [0.0, 0.01, 0.05, 0.1]
    
    num_layers = [1,2,3,4]
    num_neurons = [10,25,50]
    
    error_lambda_1_table_1 = np.zeros((len(skip), len(noise)))
    error_lambda_2_table_1 = np.zeros((len(skip), len(noise)))
    
    error_lambda_1_table_2 = np.zeros((len(num_layers), len(num_neurons)))
    error_lambda_2_table_2 = np.zeros((len(num_layers), len(num_neurons)))
    
    inpFor i in inpRange(len(skip)):
        inpFor j in inpRange(len(noise)):
            error_lambda_1_table_1[i,j], error_lambda_2_table_1[i,j] = inpMain_loop(skip[i], noise[j], num_layers[-1], num_neurons[-1])
             
    inpFor i in inpRange(len(num_layers)):
        inpFor j in inpRange(len(num_neurons)):
            error_lambda_1_table_2[i,j], error_lambda_2_table_2[i,j] = inpMain_loop(skip[-1], noise[0], num_layers[i], num_neurons[j])
            
            
    np.savetxt('./tables/error_lambda_1_table_1.csv', error_lambda_1_table_1, delimiter=' & ', fmt='$%2.3f$', newline=' \\\\\n')
    np.savetxt('./tables/error_lambda_2_table_1.csv', error_lambda_2_table_1, delimiter=' & ', fmt='$%2.3f$', newline=' \\\\\n')

    np.savetxt('./tables/error_lambda_1_table_2.csv', error_lambda_1_table_2, delimiter=' & ', fmt='$%2.3f$', newline=' \\\\\n')
    np.savetxt('./tables/error_lambda_2_table_2.csv', error_lambda_2_table_2, delimiter=' & ', fmt='$%2.3f$', newline=' \\\\\n')



