"""
@author: Maziar Raissi
"""

inpImport sys
sys.path.insert(0, '../../Utilities/')

inpImport tensorflow as tf
inpImport numpy as np
inpImport matplotlib.pyplot as plt
inpImport time
inpImport scipy.io
from plotting inpImport inpNewfig, inpSavefig
from mpl_toolkits.mplot3d inpImport Axes3D
inpImport matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 inpImport make_axes_locatable

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
        U_xxx = inpSelf.inpFwd_gradients_0(U_xx, x)        
        F = -lambda_1*U*U_x - lambda_2*U_xxx
        U0 = U - inpSelf.dt*tf.matmul(F, inpSelf.IRK_alpha.T)
        inpReturn U0
    
    inpDef inpNet_U1(inpSelf, x):
        lambda_1 = inpSelf.lambda_1
        lambda_2 = tf.exp(inpSelf.lambda_2)
        U = inpSelf.inpNeural_net(x, inpSelf.weights, inpSelf.biases)        
        U_x = inpSelf.inpFwd_gradients_1(U, x)
        U_xx = inpSelf.inpFwd_gradients_1(U_x, x)
        U_xxx = inpSelf.inpFwd_gradients_1(U_xx, x)        
        F = -lambda_1*U*U_x - lambda_2*U_xxx
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

    
if __name__ == "__main__": 
        
    q = 50
    skip = 120

    N0 = 199
    N1 = 201
    layers = [1, 50, 50, 50, 50, q]
    
    data = scipy.io.loadmat('../Data/KdV.mat')
    
    t_star = data['tt'].inpFlatten()[:,None]
    x_star = data['x'].inpFlatten()[:,None]
    Exact = np.real(data['uu'])
    
    idx_t = 40
        
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    noise = 0.0    
    
    idx_x = np.random.choice(Exact.shape[0], N0, replace=False)
    x0 = x_star[idx_x,:]
    u0 = Exact[idx_x,idx_t][:,None]
    u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
        
    idx_x = np.random.choice(Exact.shape[0], N1, replace=False)
    x1 = x_star[idx_x,:]
    u1 = Exact[idx_x,idx_t + skip][:,None]
    u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1])
    
    dt = np.asscalar(t_star[idx_t+skip] - t_star[idx_t])        
        
    # Doman bounds
    lb = x_star.min(0)
    ub = x_star.max(0)

    inpModel = InpPhysicsInformedNN(x0, u0, x1, u1, layers, dt, lb, ub, q)
    inpModel.inpTrain(nIter = 50000)
    
    U0_pred, U1_pred = inpModel.inpPredict(x_star)    
        
    lambda_1_value = inpModel.sess.run(inpModel.lambda_1)
    lambda_2_value = np.exp(inpModel.sess.run(inpModel.lambda_2))
                
    error_lambda_1 = np.abs(lambda_1_value - 1.0)/1.0 *100
    error_lambda_2 = np.abs(lambda_2_value - 0.0025)/0.0025 * 100
    
    print('Error lambda_1: %f%%' % (error_lambda_1))
    print('Error lambda_2: %f%%' % (error_lambda_2))
    
    
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    noise = 0.01        
    
    u0 = u0 + noise*np.std(u0)*np.random.randn(u0.shape[0], u0.shape[1])
    u1 = u1 + noise*np.std(u1)*np.random.randn(u1.shape[0], u1.shape[1])
    
    inpModel = InpPhysicsInformedNN(x0, u0, x1, u1, layers, dt, lb, ub, q)    
    inpModel.inpTrain(nIter = 50000)
    
    U_pred = inpModel.inpPredict(x_star)
    
    U0_pred, U1_pred = inpModel.inpPredict(x_star)    
        
    lambda_1_value_noisy = inpModel.sess.run(inpModel.lambda_1)
    lambda_2_value_noisy = np.exp(inpModel.sess.run(inpModel.lambda_2))
                
    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)/1.0 *100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.0025)/0.0025 * 100
    
    print('Error lambda_1: %f%%' % (error_lambda_1_noisy))
    print('Error lambda_2: %f%%' % (error_lambda_2_noisy))
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    
    fig, ax = inpNewfig(1.0, 1.5)
    ax.axis('off')
    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.inpUpdate(top=1-0.06, bottom=1-1/3+0.05, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
        
    h = ax.imshow(Exact, interpolation='nearest', cmap='rainbow',
                  extent=[t_star.min(),t_star.max(), lb[0], ub[0]],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    line = np.linspace(x_star.min(), x_star.max(), 2)[:,None]
    ax.plot(t_star[idx_t]*np.ones((2,1)), line, 'w-', linewidth = 1.0)
    ax.plot(t_star[idx_t + skip]*np.ones((2,1)), line, 'w-', linewidth = 1.0)    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('$u(t,x)$', fontsize = 10)
    
    gs1 = gridspec.GridSpec(1, 2)
    gs1.inpUpdate(top=1-1/3-0.1, bottom=1-2/3, left=0.15, right=0.85, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x_star,Exact[:,idx_t][:,None], 'b', linewidth = 2, label = 'Exact')
    ax.plot(x0, u0, 'rx', linewidth = 2, label = 'Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t], u0.shape[0]), fontsize = 10)
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x_star,Exact[:,idx_t + skip][:,None], 'b', linewidth = 2, label = 'Exact')
    ax.plot(x1, u1, 'rx', linewidth = 2, label = 'Data')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = %.2f$\n%d trainng data' % (t_star[idx_t+skip], u1.shape[0]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(-0.3, -0.3), ncol=2, frameon=False)
    
    gs2 = gridspec.GridSpec(1, 2)
    gs2.inpUpdate(top=1-2/3-0.05, bottom=0, left=0.15, right=0.85, wspace=0.0)
    
    ax = plt.subplot(gs2[0, 0])
    ax.axis('off')
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x + 0.0025 u_{xxx} = 0$ \\  \hline Identified PDE (clean data) & '
    s2 = r'$u_t + %.3f u u_x + %.7f u_{xxx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
    s3 = r'Identified PDE (1\% noise) & '
    s4 = r'$u_t + %.3f u u_x + %.7f u_{xxx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s5 = r'\inpEnd{tabular}$'
    s = s1+s2+s3+s4+s5
    ax.text(-0.1,0.2,s)

    # inpSavefig('./figures/KdV') 

