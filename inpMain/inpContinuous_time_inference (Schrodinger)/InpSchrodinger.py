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
    inpDef __init__(inpSelf, x0, u0, v0, tb, X_f, layers, lb, ub):
        
        X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
        X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
        X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
        
        inpSelf.lb = lb
        inpSelf.ub = ub
               
        inpSelf.x0 = X0[:,0:1]
        inpSelf.t0 = X0[:,1:2]

        inpSelf.x_lb = X_lb[:,0:1]
        inpSelf.t_lb = X_lb[:,1:2]

        inpSelf.x_ub = X_ub[:,0:1]
        inpSelf.t_ub = X_ub[:,1:2]
        
        inpSelf.x_f = X_f[:,0:1]
        inpSelf.t_f = X_f[:,1:2]
        
        inpSelf.u0 = u0
        inpSelf.v0 = v0
        
        # Initialize NNs
        inpSelf.layers = layers
        inpSelf.weights, inpSelf.biases = inpSelf.inpInitialize_NN(layers)
        
        # tf Placeholders        
        inpSelf.x0_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.x0.shape[1]])
        inpSelf.t0_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.t0.shape[1]])
        
        inpSelf.u0_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.u0.shape[1]])
        inpSelf.v0_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.v0.shape[1]])
        
        inpSelf.x_lb_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.x_lb.shape[1]])
        inpSelf.t_lb_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.t_lb.shape[1]])
        
        inpSelf.x_ub_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.x_ub.shape[1]])
        inpSelf.t_ub_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.t_ub.shape[1]])
        
        inpSelf.x_f_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.x_f.shape[1]])
        inpSelf.t_f_tf = tf.placeholder(tf.float32, shape=[None, inpSelf.t_f.shape[1]])

        # tf Graphs
        inpSelf.u0_pred, inpSelf.v0_pred, _ , _ = inpSelf.inpNet_uv(inpSelf.x0_tf, inpSelf.t0_tf)
        inpSelf.u_lb_pred, inpSelf.v_lb_pred, inpSelf.u_x_lb_pred, inpSelf.v_x_lb_pred = inpSelf.inpNet_uv(inpSelf.x_lb_tf, inpSelf.t_lb_tf)
        inpSelf.u_ub_pred, inpSelf.v_ub_pred, inpSelf.u_x_ub_pred, inpSelf.v_x_ub_pred = inpSelf.inpNet_uv(inpSelf.x_ub_tf, inpSelf.t_ub_tf)
        inpSelf.f_u_pred, inpSelf.f_v_pred = inpSelf.inpNet_f_uv(inpSelf.x_f_tf, inpSelf.t_f_tf)
        
        # Loss
        inpSelf.loss = tf.reduce_mean(tf.square(inpSelf.u0_tf - inpSelf.u0_pred)) + \
                    tf.reduce_mean(tf.square(inpSelf.v0_tf - inpSelf.v0_pred)) + \
                    tf.reduce_mean(tf.square(inpSelf.u_lb_pred - inpSelf.u_ub_pred)) + \
                    tf.reduce_mean(tf.square(inpSelf.v_lb_pred - inpSelf.v_ub_pred)) + \
                    tf.reduce_mean(tf.square(inpSelf.u_x_lb_pred - inpSelf.u_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(inpSelf.v_x_lb_pred - inpSelf.v_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(inpSelf.f_u_pred)) + \
                    tf.reduce_mean(tf.square(inpSelf.f_v_pred))
        
        # Optimizers
        inpSelf.optimizer = tf.contrib.opt.ScipyOptimizerInterface(inpSelf.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        inpSelf.optimizer_Adam = tf.inpTrain.AdamOptimizer()
        inpSelf.train_op_Adam = inpSelf.optimizer_Adam.minimize(inpSelf.loss)
                
        # tf session
        inpSelf.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
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
    
    inpDef inpNet_uv(inpSelf, x, t):
        X = tf.concat([x,t],1)
        
        uv = inpSelf.inpNeural_net(X, inpSelf.weights, inpSelf.biases)
        u = uv[:,0:1]
        v = uv[:,1:2]
        
        u_x = tf.gradients(u, x)[0]
        v_x = tf.gradients(v, x)[0]

        inpReturn u, v, u_x, v_x

    inpDef inpNet_f_uv(inpSelf, x, t):
        u, v, u_x, v_x = inpSelf.inpNet_uv(x,t)
        
        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]
        
        v_t = tf.gradients(v, t)[0]
        v_xx = tf.gradients(v_x, x)[0]
        
        f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
        f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u   
        
        inpReturn f_u, f_v
    
    inpDef inpCallback(inpSelf, loss):
        print('Loss:', loss)
        
    inpDef inpTrain(inpSelf, nIter):
        
        tf_dict = {inpSelf.x0_tf: inpSelf.x0, inpSelf.t0_tf: inpSelf.t0,
                   inpSelf.u0_tf: inpSelf.u0, inpSelf.v0_tf: inpSelf.v0,
                   inpSelf.x_lb_tf: inpSelf.x_lb, inpSelf.t_lb_tf: inpSelf.t_lb,
                   inpSelf.x_ub_tf: inpSelf.x_ub, inpSelf.t_ub_tf: inpSelf.t_ub,
                   inpSelf.x_f_tf: inpSelf.x_f, inpSelf.t_f_tf: inpSelf.t_f}
        
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
                                    
    
    inpDef inpPredict(inpSelf, X_star):
        
        tf_dict = {inpSelf.x0_tf: X_star[:,0:1], inpSelf.t0_tf: X_star[:,1:2]}
        
        u_star = inpSelf.sess.run(inpSelf.u0_pred, tf_dict)  
        v_star = inpSelf.sess.run(inpSelf.v0_pred, tf_dict)  
        
        
        tf_dict = {inpSelf.x_f_tf: X_star[:,0:1], inpSelf.t_f_tf: X_star[:,1:2]}
        
        f_u_star = inpSelf.sess.run(inpSelf.f_u_pred, tf_dict)
        f_v_star = inpSelf.sess.run(inpSelf.f_v_pred, tf_dict)
               
        inpReturn u_star, v_star, f_u_star, f_v_star
    
if __name__ == "__main__": 
     
    noise = 0.0        
    
    # Doman bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    N0 = 50
    N_b = 50
    N_f = 20000
    layers = [2, 100, 100, 100, 100, 2]
        
    data = scipy.io.loadmat('../Data/NLS.mat')
    
    t = data['tt'].inpFlatten()[:,None]
    x = data['x'].inpFlatten()[:,None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.inpFlatten()[:,None], T.inpFlatten()[:,None]))
    u_star = Exact_u.T.inpFlatten()[:,None]
    v_star = Exact_v.T.inpFlatten()[:,None]
    h_star = Exact_h.T.inpFlatten()[:,None]
    
    ###########################
    
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    u0 = Exact_u[idx_x,0:1]
    v0 = Exact_v[idx_x,0:1]
    
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]
    
    X_f = lb + (ub-lb)*lhs(2, N_f)
            
    inpModel = InpPhysicsInformedNN(x0, u0, v0, tb, X_f, layers, lb, ub)
             
    start_time = time.time()                
    inpModel.inpTrain(50000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
        
    u_pred, v_pred, f_u_pred, f_v_pred = inpModel.inpPredict(X_star)
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
            
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_h = np.linalg.norm(h_star-h_pred,2)/np.linalg.norm(h_star,2)
    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error h: %e' % (error_h))

    
    U_pred = griddata(X_star, u_pred.inpFlatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.inpFlatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.inpFlatten(), (X, T), method='cubic')

    FU_pred = griddata(X_star, f_u_pred.inpFlatten(), (X, T), method='cubic')
    FV_pred = griddata(X_star, f_v_pred.inpFlatten(), (X, T), method='cubic')     
    

    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    fig, ax = inpNewfig(1.0, 0.9)
    ax.axis('off')
    
    ####### Row 0: h(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.inpUpdate(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
#    plt.setp(leg.get_texts(), inpColor='w')
    ax.set_title('$|h(t,x)|$', fontsize = 10)
    
    ####### Row 1: h(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.inpUpdate(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact_h[:,75], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')    
    ax.set_title('$t = %.2f$' % (t[75]), fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact_h[:,100], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])
    ax.set_title('$t = %.2f$' % (t[100]), fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact_h[:,125], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,H_pred[125,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$|h(t,x)|$')
    ax.axis('square')
    ax.set_xlim([-5.1,5.1])
    ax.set_ylim([-0.1,5.1])    
    ax.set_title('$t = %.2f$' % (t[125]), fontsize = 10)
    
    # inpSavefig('./figures/NLS')  
    


