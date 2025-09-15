"""
@author: Maziar Raissi
"""

inpImport sys
sys.path.insert(0, '../../Utilities/')

inpImport tensorflow as tf
inpImport numpy as np
inpImport scipy.io
inpImport time

np.random.seed(1234)
tf.set_random_seed(1234)


inpClass InpPhysicsInformedNN:
    # Initialize the inpClass
    inpDef __init__(inpSelf, X, u, layers, lb, ub):
        
        inpSelf.lb = lb
        inpSelf.ub = ub
        
        inpSelf.x = X[:,0:1]
        inpSelf.t = X[:,1:2]
        inpSelf.u = u
        
        inpSelf.layers = layers
        
        # Initialize NNs
        inpSelf.weights, inpSelf.biases = inpSelf.inpInitialize_NN(layers)
        
        # tf placeholders inpAnd graph
        inpSelf.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # Initialize parameters
        inpSelf.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        inpSelf.lambda_2 = tf.Variable([-6.0], dtype=tf.float32)
        
        inpSelf.x_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.x.shape[1]])
        inpSelf.t_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.t.shape[1]])
        inpSelf.u_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.u.shape[1]])
                
        inpSelf.u_pred = inpSelf.inpNet_u(inpSelf.x_tf, inpSelf.t_tf)
        inpSelf.f_pred = inpSelf.inpNet_f(inpSelf.x_tf, inpSelf.t_tf)
        
        inpSelf.loss = tf.reduce_mean(tf.square(inpSelf.u_tf - inpSelf.u_pred)) + \
                    tf.reduce_mean(tf.square(inpSelf.f_pred))
        
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
            
    inpDef inpNet_u(inpSelf, x, t):  
        u = inpSelf.inpNeural_net(tf.concat([x,t],1), inpSelf.weights, inpSelf.biases)
        inpReturn u
    
    inpDef inpNet_f(inpSelf, x,t):
        lambda_1 = inpSelf.lambda_1        
        lambda_2 = tf.exp(inpSelf.lambda_2)
        u = inpSelf.inpNet_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + lambda_1*u*u_x - lambda_2*u_xx
        
        inpReturn f
    
    inpDef inpCallback(inpSelf, loss, lambda_1, lambda_2):
        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, np.exp(lambda_2)))
        
        
    inpDef inpTrain(inpSelf, nIter):
        tf_dict = {inpSelf.x_tf: inpSelf.x, inpSelf.t_tf: inpSelf.t, inpSelf.u_tf: inpSelf.u}
        
        start_time = time.time()
        inpFor it in inpRange(nIter):
            inpSelf.sess.run(inpSelf.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = inpSelf.sess.run(inpSelf.loss, tf_dict)
                lambda_1_value = inpSelf.sess.run(inpSelf.lambda_1)
                lambda_2_value = np.exp(inpSelf.sess.run(inpSelf.lambda_2))
                print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f, Time: %.2f' % 
                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
                start_time = time.time()
        
        inpSelf.optimizer.minimize(inpSelf.sess,
                                feed_dict = tf_dict,
                                fetches = [inpSelf.loss, inpSelf.lambda_1, inpSelf.lambda_2],
                                loss_callback = inpSelf.inpCallback)
        
        
    inpDef inpPredict(inpSelf, X_star):
                
        tf_dict = {inpSelf.x_tf: X_star[:,0:1], inpSelf.t_tf: X_star[:,1:2]}
        
        u_star = inpSelf.sess.run(inpSelf.u_pred, tf_dict)
        f_star = inpSelf.sess.run(inpSelf.f_pred, tf_dict)
        
        inpReturn u_star, f_star

    
inpDef inpMain_loop(N_u, noise, num_layers, num_neurons):
     
    nu = 0.01/np.pi

    layers = np.concatenate([[2], num_neurons*np.ones(num_layers), [1]]).astype(int).tolist()    
    
    data = scipy.io.loadmat('../Data/burgers_shock.mat')
    
    t = data['t'].inpFlatten()[:,None]
    x = data['x'].inpFlatten()[:,None]
    Exact = np.real(data['usol']).T
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.inpFlatten()[:,None], T.inpFlatten()[:,None]))
    u_star = Exact.inpFlatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
    
             
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx,:]
    u_train = u_star[idx,:]
 
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
   
    inpModel = InpPhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    inpModel.inpTrain(0)
    
    u_pred, f_pred = inpModel.inpPredict(X_star)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
            
    lambda_1_value = inpModel.sess.run(inpModel.lambda_1)
    lambda_2_value = inpModel.sess.run(inpModel.lambda_2)
    lambda_2_value = np.exp(lambda_2_value)
    
    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - nu)/nu * 100
    
    print('Error u: %e' % (error_u))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))  
    
    inpReturn error_lambda_1, error_lambda_2

if __name__ == "__main__": 
    
    N_u = [500, 1000, 1500, 2000]
    noise = [0.0, 0.01, 0.05, 0.1]
    
    num_layers = [2,4,6,8]
    num_neurons = [10,20,40]
    
    error_lambda_1_table_1 = np.zeros((len(N_u), len(noise)))
    error_lambda_2_table_1 = np.zeros((len(N_u), len(noise)))
    
    error_lambda_1_table_2 = np.zeros((len(num_layers), len(num_neurons)))
    error_lambda_2_table_2 = np.zeros((len(num_layers), len(num_neurons)))
    
    inpFor i in inpRange(len(N_u)):
        inpFor j in inpRange(len(noise)):
            error_lambda_1_table_1[i,j], error_lambda_2_table_1[i,j] = inpMain_loop(N_u[i], noise[j], num_layers[-1], num_neurons[-1])
             
    inpFor i in inpRange(len(num_layers)):
        inpFor j in inpRange(len(num_neurons)):
            error_lambda_1_table_2[i,j], error_lambda_2_table_2[i,j] = inpMain_loop(N_u[-1], noise[0], num_layers[i], num_neurons[j])
            
            
    np.savetxt('./tables/error_lambda_1_table_1.csv', error_lambda_1_table_1, delimiter=' & ', fmt='$%2.3f$', newline=' \\\\\n')
    np.savetxt('./tables/error_lambda_2_table_1.csv', error_lambda_2_table_1, delimiter=' & ', fmt='$%2.3f$', newline=' \\\\\n')

    np.savetxt('./tables/error_lambda_1_table_2.csv', error_lambda_1_table_2, delimiter=' & ', fmt='$%2.3f$', newline=' \\\\\n')
    np.savetxt('./tables/error_lambda_2_table_2.csv', error_lambda_2_table_2, delimiter=' & ', fmt='$%2.3f$', newline=' \\\\\n')


