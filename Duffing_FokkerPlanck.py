#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wayne Isaac Tan Uy, PhD

Code here is based on the PINN code by Maziar Raissi which can be found at
https://github.com/maziarraissi/PINNs

This code is example 5.3.1 in the manuscript.

"""

import tensorflow as tf
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import griddata
from pyDOE import lhs
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X0, InitCond, Xf, Nf_x, Nf_t, layers, lb, ub, dampRatio, nu, alfa, intensity):

        # Don't have boundary conditions 
        # X0 are points along the spatial dimension at which we enforce the initial conditions
        # InitCond are the initial condition values
        # Xf are the collocation points at which to enforce the defining equation
        # Nf_x, Nf_t - number of collocation nodes in space, time to enforce operator
        # layers: [input layer hidden layers output layer]
        # lb, ub: lower and upper bounds for x,t
        # dampRatio, nu, alfa, intensity are parameters of the system
        
        self.dampRatio = dampRatio
        self.nu = nu
        self.alfa = alfa
        self.intensity = intensity

        # Initial condition

        self.x10 = X0[:,0:1]
        self.x20 = X0[:,1:2]
        self.t0 = X0[:,2:3]
        self.u0 = InitCond
        
        # PDE conditions
        
        self.x1f = Xf[:,0:1]
        self.x2f = Xf[:,1:2]
        self.tf = Xf[:,2:3]
        
        self.lb = lb
        self.ub = ub
        
        self.Nf_x = Nf_x
        self.Nf_t = Nf_t
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Placeholders       
        # _tf signifies that these are needed for the graph of tensorflow
        
        self.x10_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x20_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.x1f_tf = tf.placeholder(tf.float32, shape=[self.x1f.shape[0], 1])
        self.x2f_tf = tf.placeholder(tf.float32, shape=[self.x2f.shape[0], 1])
        self.tf_tf = tf.placeholder(tf.float32, shape=[self.tf.shape[0], 1])
        
        # tf Graphs
        # Graph for initial condition
        self.u0_pred, self.u0_t_pred = self.net_u(self.x10_tf, self.x20_tf, self.t0_tf)
        
        # Graph to enforce pde
        self.f_u_pred = self.net_f(self.x1f_tf, self.x2f_tf, self.tf_tf) 
        
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
                    tf.reduce_mean(tf.square(self.f_u_pred)) 
        
        # Optimizers
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 100000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
                
        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
              
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def net_u(self, x1, x2, t):
        Xtemp = tf.concat([x1,x2],1)
        X = tf.concat([Xtemp,t],1)
        
        uNN = self.neural_net(X, self.weights, self.biases)
        uNN_t = tf.gradients(uNN, t)[0]    
        
        return uNN, uNN_t

    def net_f(self, x1, x2, t):
        
        # compute u, u_t
        u, u_t = self.net_u(x1, x2, t)
        
        # compute Volume
        Vol = (self.ub[1] - self.lb[1])*(self.ub[0] - self.lb[0])
        
        # reshape u, u_t
        u_rsh = tf.reshape(u,[self.Nf_t,self.Nf_x])
        u_t_rsh = tf.reshape(u_t,[self.Nf_t,self.Nf_x])
        expMinusu = tf.exp(-1.0*u_rsh)

        # compute integral of numerator
        numerIntgrnd = tf.multiply(expMinusu,u_t_rsh)
        numerInteg = Vol*tf.reduce_mean(numerIntgrnd,1)
        
        # compute integral of denominator
        denomInteg = Vol*tf.reduce_mean(expMinusu,1)

        # compute quotient
        quotInteg = tf.divide(numerInteg,denomInteg)
        
        # reshape quotient to get intVals
        quotInteg = tf.reshape(quotInteg,[self.Nf_t,1])
        intVals = tf.tile(quotInteg,[1,self.Nf_x])
        intVals = tf.reshape(intVals,[self.Nf_t*self.Nf_x,1])
        
        u_x1 = tf.gradients(u, x1)[0]
        u_x2 = tf.gradients(u, x2)[0]
        u_x2x2 = tf.gradients(u_x2, x2)[0]
        a2 = -self.nu**2*(x1 + self.alfa*x1**3) - 2.0*self.dampRatio*self.nu*x2
        
        f_u = u_t + x2*u_x1 + 2*self.dampRatio*self.nu + a2*u_x2 + 0.5*np.pi*self.intensity*(u_x2**2 - u_x2x2) - intVals 
                
        return f_u
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        
        tf_dict = {self.x10_tf: self.x10, self.x20_tf: self.x20, self.t0_tf: self.t0, self.u0_tf: self.u0,
                   self.x1f_tf: self.x1f, self.x2f_tf: self.x2f, self.tf_tf: self.tf}
              
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = time.time()
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
                                    
    
    def predict(self, X_star):
        
        # X_star points (u,v,t) at which the solution is evaluated, 
        
        tf_dict = {self.x10_tf: X_star[:,0:1], self.x20_tf: X_star[:,1:2], self.t0_tf: X_star[:,2:3]}
        
        u_star = self.sess.run(self.u0_pred, tf_dict)        
        
        return u_star
    
    
if __name__ == "__main__": 
    
    noise = 0.0        
    
    # declare parameters
    dampRatio = 0.25
    sigma1 = 1;
    sigma2 = 1;
    corrParam = 0.8;
    nu = 1;
    alfa = 1;
    intensity = 1;
    
    # Domain bounds
    lb = np.array([-4.0, -8.0, 0.0]) # x1, x2, t bounds
    ub = np.array([ 4.0,  8.0, 1.0])

    Nf_x = 1000 # satisfying the governing equations
    Nf_t = 50# satisfying governing equations
    
    layers = [3, 50, 50, 50, 50, 50, 50, 1]
    
    # for prediction
    tPred = np.array([0., 0.25, 0.5, 0.75, 1.])
    tPred = tPred[:,None] 
    x1Pred = np.linspace(lb[0],ub[0],161)
    x2Pred = np.linspace(lb[1],ub[1],321)
    x1Pred = x1Pred[:,None]
    x2Pred = x2Pred[:,None]
    
    x1PredMesh,x2PredMesh = np.meshgrid(x1Pred,x2Pred)
    X1X2Rsh = np.hstack((x1PredMesh.flatten()[:,None], x2PredMesh.flatten()[:,None]))
    
    # training data
    
    # collocation points for initial condition
    x1Init = np.linspace(lb[0],ub[0],25)
    x2Init = np.linspace(lb[1],ub[1],41)
    
    N0 = x1Init.shape[0]*x2Init.shape[0]
    
    x1MeshInit, x2MeshInit = np.meshgrid(x1Init,x2Init) 
    XwoT = np.hstack((x1MeshInit.flatten()[:,None], x2MeshInit.flatten()[:,None]))
    X0 = np.hstack((XwoT,np.zeros((N0,1))))
    InitCond = 0.5*(1.0/(1-corrParam**2))*(XwoT[:,0:1]**2 +XwoT[:,1:2]**2 - 2.0*corrParam*XwoT[:,0:1]*XwoT[:,1:2])
       
    # collocation points to enforce PDE    
 
    Xf_x = lb[0:2] + (ub[0:2]-lb[0:2])*lhs(2,Nf_x)
    Xf_t = lb[-1] + (ub[-1]-lb[-1])*lhs(1,Nf_t)
    
    Xf = np.empty([0,3])
    
    for timeID in range(Xf_t.shape[0]):

        tempCollocNodes = np.hstack((Xf_x,Xf_t[timeID]*np.ones([Nf_x,1])))
        
        Xf = np.vstack((Xf,tempCollocNodes))
     
    # create model
    model = PhysicsInformedNN(X0, InitCond, Xf, Nf_x, Nf_t, layers, lb, ub, dampRatio, nu, alfa, intensity)
    
    start_time = time.time()                
    model.train(25000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    # Compute predicted characteristic function
   
    fPred = np.zeros((x2Pred.shape[0],x1Pred.shape[0],tPred.shape[0]))
    
    for timeID in range(tPred.shape[0]):
        # form X_star to include t
        tSlice = tPred[timeID]*np.ones(x1PredMesh.shape)
        X_star = np.hstack((X1X2Rsh, tSlice.flatten()[:,None]))
        
        # perform prediction
        u_pred_t = model.predict(X_star)
        
        # reshape
        u_pred_rsh = np.reshape(u_pred_t,(x2Pred.shape[0],x1Pred.shape[0]))
        
        # normalize
        
        expNN = np.exp(-1.0*u_pred_rsh)
        normFactor = np.trapz(expNN,x=np.transpose(x1Pred),axis = 1)
        normFactor = normFactor[:,None]
        normFactor = np.trapz(normFactor,x=x2Pred,axis=0)
    
        fPred[:,:,timeID] = expNN/normFactor
        
    #%%
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    

    plt.rc('text',usetex=False) 
    
   # plot the estimated pdf
    
    fig = plt.figure(figsize=(10,15))
    
    ############## Error at time slices ##################
    ax = plt.subplot(3,2,1)
    
    h = ax.imshow(fPred[:,:,0], interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[0], ub[0], lb[1], ub[1]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax, orientation='vertical')
    plt.xlabel('x1')
    plt.ylabel('x2')    
    plt.title('t = %.2f' % (tPred[0]), fontsize = 10)
    
    ax = plt.subplot(3,2,2)
    
    h = ax.imshow(fPred[:,:,1], interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[0], ub[0], lb[1], ub[1]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax, orientation='vertical')
    plt.xlabel('x1')
    plt.ylabel('x2')    
    plt.title('t = %.2f' % (tPred[1]), fontsize = 10)
    
    ax = plt.subplot(3,2,3)
    
    h = ax.imshow(fPred[:,:,2], interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[0], ub[0], lb[1], ub[1]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax, orientation='vertical')
    plt.xlabel('x1')
    plt.ylabel('x2')    
    plt.title('t = %.2f' % (tPred[2]), fontsize = 10)
    
    ax = plt.subplot(3,2,4)
    
    h = ax.imshow(fPred[:,:,3], interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[0], ub[0], lb[1], ub[1]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax, orientation='vertical')
    plt.xlabel('x1')
    plt.ylabel('x2')    
    plt.title('t = %.2f' % (tPred[3]), fontsize = 10)
    
    ax = plt.subplot(3,2,5.5)
    
    h = ax.imshow(fPred[:,:,4], interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[0], ub[0], lb[1], ub[1]], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax, orientation='vertical')
    plt.xlabel('x1')
    plt.ylabel('x2')    
    plt.title('t = %.2f' % (tPred[4]), fontsize = 10)
    
    plt.show()
