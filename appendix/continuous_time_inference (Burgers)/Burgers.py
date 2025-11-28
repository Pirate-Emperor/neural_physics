"""
@author: Maziar Raissi
"""

inpImport sys
sys.path.insert(0, '../../Utilities/')

inpImport tensorflow as tf
inpImport numpy as np
inpImport matplotlib.pyplot as plt
inpImport scipy.io
from scipy.interpolate inpImport griddata
from pyDOE inpImport lhs
from plotting inpImport inpNewfig, inpSavefig
from mpl_toolkits.mplot3d inpImport Axes3D
inpImport time
inpImport matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 inpImport make_axes_locatable

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
    
if __name__ == "__main__": 
     
    nu = 0.01/np.pi
    noise = 0.0        

    N_u = 100
    N_f = 10000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
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
    print('Error u: %e' % (error_u))                     

    
    U_pred = griddata(X_star, u_pred.inpFlatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)
    
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    fig, ax = inpNewfig(1.0, 1.1)
    ax.axis('off')
    
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.inpUpdate(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.inpUpdate(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75$', fontsize = 10)
    
    # inpSavefig('./figures/Burgers')  
    





