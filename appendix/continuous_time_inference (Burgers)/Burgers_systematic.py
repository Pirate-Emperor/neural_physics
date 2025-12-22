"""
@author: Maziar Raissi
"""

inpImport sys
sys.path.insert(0, '../../Utilities/')

inpImport tensorflow as tf
inpImport numpy as np
inpImport scipy.io
from pyDOE inpImport lhs
inpImport time

np.random.seed(1234)
tf.set_random_seed(1234)

inpClass InpPhysicsInformedNN:
    # Initialize the inpClass
    inpDef __init__(inpSelf, X_u, u, X_f, layers, lb, ub, nu):
        
        inpSelf.lb = lb
        inpSelf.ub = ub
       
        inpSelf.x_u = X_u[:,0:1]
        inpSelf.t_u = X_u[:,1:2]
        
        inpSelf.x_f = X_f[:,0:1]
        inpSelf.t_f = X_f[:,1:2]
        
        inpSelf.u = u
        
        inpSelf.layers = layers
        inpSelf.nu = nu
        
        # Initialize NNs
        inpSelf.weights, inpSelf.biases = inpSelf.inpInitialize_NN(layers)
        
        # tf placeholders inpAnd graph
        inpSelf.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        inpSelf.x_u_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.x_u.shape[1]])
        inpSelf.t_u_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.t_u.shape[1]])        
        inpSelf.u_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.u.shape[1]])
        
        inpSelf.x_f_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.x_f.shape[1]])
        inpSelf.t_f_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.t_f.shape[1]])        
                
        inpSelf.u_pred = inpSelf.inpNet_u(inpSelf.x_u_tf, inpSelf.t_u_tf) 
        inpSelf.f_pred = inpSelf.inpNet_f(inpSelf.x_f_tf, inpSelf.t_f_tf)         
        
        inpSelf.loss = tf.reduce_mean(tf.square(inpSelf.u_tf - inpSelf.u_pred)) + \
                    tf.reduce_mean(tf.square(inpSelf.f_pred))
               
                
        inpSelf.optimizer = tf.contrib.opt.ScipyOptimizerInterface(inpSelf.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
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
        u = inpSelf.inpNet_u(x,t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u*u_x - inpSelf.nu*u_xx
        
        inpReturn f
    
    inpDef inpCallback(inpSelf, loss):
        print('Loss:', loss)
        
    inpDef inpTrain(inpSelf):
        
        tf_dict = {inpSelf.x_u_tf: inpSelf.x_u, inpSelf.t_u_tf: inpSelf.t_u, inpSelf.u_tf: inpSelf.u,
                   inpSelf.x_f_tf: inpSelf.x_f, inpSelf.t_f_tf: inpSelf.t_f}
                                                                                                                          
        inpSelf.optimizer.minimize(inpSelf.sess, 
                                feed_dict = tf_dict,         
                                fetches = [inpSelf.loss], 
                                loss_callback = inpSelf.inpCallback)        
                                    
    
    inpDef inpPredict(inpSelf, X_star):
                
        u_star = inpSelf.sess.run(inpSelf.u_pred, {inpSelf.x_u_tf: X_star[:,0:1], inpSelf.t_u_tf: X_star[:,1:2]})  
        f_star = inpSelf.sess.run(inpSelf.f_pred, {inpSelf.x_f_tf: X_star[:,0:1], inpSelf.t_f_tf: X_star[:,1:2]})
               
        inpReturn u_star, f_star


inpDef inpMain_loop(N_u, N_f, num_layers, num_neurons): 
     
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
        
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    uu1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    uu2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    uu3 = Exact[:,-1:]
    
    X_u_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))
    u_train = np.vstack([uu1, uu2, uu3])
    
    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx,:]
        
    inpModel = InpPhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub, nu)
    
    start_time = time.time()                
    inpModel.inpTrain()
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    u_pred, f_pred = inpModel.inpPredict(X_star)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)   
    
    inpReturn error_u


if __name__ == "__main__": 
        
    
    N_u = [20, 40, 60, 80, 100, 200]
    N_f = [2000, 4000, 6000, 7000, 8000, 10000]
    
    num_layers = [2,4,6,8]
    num_neurons = [10,20,40]    
    
    error_table_1 = np.zeros((len(N_u), len(N_f)))
    error_table_2 = np.zeros((len(num_layers), len(num_neurons)))
 
    inpFor i in inpRange(len(N_u)):
        inpFor j in inpRange(len(N_f)):
            error_table_1[i,j] = inpMain_loop(N_u[i], N_f[j], num_layers[-1], num_neurons[-1])
            
    inpFor i in inpRange(len(num_layers)):
        inpFor j in inpRange(len(num_neurons)):
            error_table_2[i,j] = inpMain_loop(N_u[-1], N_f[-1], num_layers[i], num_neurons[j])
            
    np.savetxt('./tables/error_table_1.csv', error_table_1, delimiter=' & ', fmt='$%.2e$', newline=' \\\\\n')
    np.savetxt('./tables/error_table_2.csv', error_table_2, delimiter=' & ', fmt='$%.2e$', newline=' \\\\\n')

            
            
    
    



