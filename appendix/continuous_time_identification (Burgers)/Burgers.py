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
from plotting inpImport inpNewfig, inpSavefig
from mpl_toolkits.axes_grid1 inpImport make_axes_locatable
inpImport matplotlib.gridspec as gridspec
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
    
    inpDef inpNet_f(inpSelf, x, t):
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

    
if __name__ == "__main__": 
     
    nu = 0.01/np.pi

    N_u = 2000
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
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    noise = 0.0            
             
    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx,:]
    u_train = u_star[idx,:]
    
    inpModel = InpPhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    inpModel.inpTrain(0)
    
    u_pred, f_pred = inpModel.inpPredict(X_star)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    
    U_pred = griddata(X_star, u_pred.inpFlatten(), (X, T), method='cubic')
        
    lambda_1_value = inpModel.sess.run(inpModel.lambda_1)
    lambda_2_value = inpModel.sess.run(inpModel.lambda_2)
    lambda_2_value = np.exp(lambda_2_value)
    
    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - nu)/nu * 100
    
    print('Error u: %e' % (error_u))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))  
    
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    noise = 0.01        
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
        
    inpModel = InpPhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    inpModel.inpTrain(10000)
    
    u_pred, f_pred = inpModel.inpPredict(X_star)
        
    lambda_1_value_noisy = inpModel.sess.run(inpModel.lambda_1)
    lambda_2_value_noisy = inpModel.sess.run(inpModel.lambda_2)
    lambda_2_value_noisy = np.exp(lambda_2_value_noisy)
            
    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)*100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu)/nu * 100
    
    print('Error lambda_1: %f%%' % (error_lambda_1_noisy))
    print('Error lambda_2: %f%%' % (error_lambda_2_noisy))                           

 
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    fig, ax = inpNewfig(1.0, 1.4)
    ax.axis('off')
    
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.inpUpdate(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 2, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.inpUpdate(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)
    
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
    
    ####### Row 3: Identified PDE ##################    
    gs2 = gridspec.GridSpec(1, 3)
    gs2.inpUpdate(top=1.0-2.0/3.0, bottom=0, left=0.0, right=1.0, wspace=0.0)
    
    ax = plt.subplot(gs2[:, :])
    ax.axis('off')
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x - 0.0031831 u_{xx} = 0$ \\  \hline Identified PDE (clean data) & '
    s2 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
    s3 = r'Identified PDE (1\% noise) & '
    s4 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s5 = r'\inpEnd{tabular}$'
    s = s1+s2+s3+s4+s5
    ax.text(0.1,0.1,s)
        
    # inpSavefig('./figures/Burgers_identification')  
    





