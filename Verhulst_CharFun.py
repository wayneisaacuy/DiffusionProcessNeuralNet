#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Wayne Isaac Tan Uy, PhD

Code here is based on the PINN code by Maziar Raissi which can be found at
https://github.com/maziarraissi/PINNs

This code is example 5.2 in the manuscript.

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
    def __init__(self, U0, InitCond, UOrigin, OriginCond, Uf, Y_int, BinaryMat, Nf_u, Nf_t, layers, lb, ub, lambdaParam, rho, a):
            
        # Don't have boundary conditions 
        # U0 are points along the spatial dimension at which we enforce the initial conditions
        # InitCond are the initial condition values
        # Uf are the collocation points at which to enforce the defining equation
        # Y_int are the nodes to compute the integral terms
        # Nf_u, Nf_t - number of collocation nodes in space, time to enforce operator
        # layers: [input layer hidden layers output layer]
        # lb, ub: lower and upper bounds for x,t
        # lambdaParam, rho, a are parameters of the system
        # BinaryMat: 2d array of 0's and 1's to enforce the char fun to be zero beyond the domain
        
        self.lambdaParam = lambdaParam
        self.rho = rho
        self.a = a

        # Initial condition

        self.u0 = U0[:,0:1]
        self.t0 = U0[:,1:2]
        self.phi0Real = InitCond[:,0:1]
        self.phi0Imag = InitCond[:,1:2]
        
        # Origin condition
        
        self.uOrigin = UOrigin[:,0:1]
        self.tOrigin = UOrigin[:,1:2]
        self.phiOrigReal = OriginCond[:,0:1]
        self.phiOrigImag = OriginCond[:,1:2]
        
        # PDE conditions
        
        self.uf = Uf[:,0:1]
        self.tf = Uf[:,1:2]
        
        self.BinaryMat = BinaryMat
        
        self.Nf_u = Nf_u
        self.Nf_t = Nf_t
    
        # Integral nodes
        
        self.yint = Y_int
        self.nInteg = self.yint.shape[0]
        
        self.lb = lb
        self.ub = ub
        
        # Initialize NNs
        self.layers = layers
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf Placeholders       
        # _tf signifies that these are needed for the graph of tensorflow
        
        self.u0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.t0_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.phi0Real_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.phi0Imag_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.uOrigin_tf = tf.placeholder(tf.float32, shape=[self.uOrigin.shape[0], 1])
        self.tOrigin_tf = tf.placeholder(tf.float32, shape=[self.uOrigin.shape[0], 1])
        self.phiOrigReal_tf = tf.placeholder(tf.float32, shape=[self.uOrigin.shape[0], 1])
        self.phiOrigImag_tf = tf.placeholder(tf.float32, shape=[self.uOrigin.shape[0], 1])
        
        self.uf_tf = tf.placeholder(tf.float32, shape=[self.uf.shape[0], 1])
        self.tf_tf = tf.placeholder(tf.float32, shape=[self.tf.shape[0], 1])
        
        self.yint_tf = tf.placeholder(tf.float32, shape=[self.yint.shape[0], 1])
        
        self.BinaryMat_tf = tf.placeholder(tf.float32, shape=[self.BinaryMat.shape[0], self.BinaryMat.shape[1]])
        
        # tf Graphs
        # Output is real and complex
        
        # Graph for initial condition
        self.phi0_pred_Real, self.phi0_pred_Imag = self.net_phi(self.u0_tf, self.t0_tf)
        
        # Graph for origin condition
        self.phiOrig_pred_Real, self.phiOrig_pred_Imag = self.net_phi(self.uOrigin_tf, self.tOrigin_tf)
        
        # Graph to enforce pde
        self.f_pred_Real, self.f_pred_Imag = self.net_f(self.uf_tf, self.tf_tf, self.yint_tf, self.BinaryMat_tf) 
        
        # Loss
        self.loss = tf.reduce_mean(tf.square(self.phi0_pred_Real - self.phi0Real_tf)) + \
                    tf.reduce_mean(tf.square(self.phi0_pred_Imag - self.phi0Imag_tf)) + \
                    tf.reduce_mean(tf.square(self.phiOrig_pred_Real - self.phiOrigReal_tf)) + \
                    tf.reduce_mean(tf.square(self.phiOrig_pred_Imag - self.phiOrigImag_tf)) + \
                    tf.reduce_mean(tf.square(self.f_pred_Real)) + tf.reduce_mean(tf.square(self.f_pred_Imag))
        
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
    
    def net_phi(self, u, t):
        
        X = tf.concat([u,t],1)
        
        phiNN = self.neural_net(X, self.weights, self.biases)
        phiNNReal = phiNN[:,0:1]
        phiNNImag = phiNN[:,1:2]
        
        return phiNNReal, phiNNImag

    def net_f(self, u, t, yint, BinaryMat):
        
        # compute phi at points where to enforce the PDE
        phiReal, phiImag = self.net_phi(u, t)
        phiReal_t = tf.gradients(phiReal, t)[0]
        phiReal_u = tf.gradients(phiReal, u)[0]
        phiImag_t = tf.gradients(phiImag, t)[0]
        phiImag_u = tf.gradients(phiImag, u)[0]
        phiReal_uu = tf.gradients(phiReal_u, u)[0]
        phiImag_uu = tf.gradients(phiImag_u, u)[0]
        
        # compute the integral term
        # compute for the first integral point
        phiReal_integ, phiImag_integ = self.net_phi(u*(1.0 + yint[0,0]) , t)
        # make adjustments for those evaluated outside the domain
        # tile BinaryMat since u,t are all collocation points while BinaryMat is just Nf_u * N_integ
        Multiplier = tf.tile(BinaryMat[:,0:1],[self.Nf_t ,1])
        phiReal_integ = tf.multiply(phiReal_integ, Multiplier)
        phiImag_integ = tf.multiply(phiImag_integ, Multiplier)
        
        # compute for the remaining integral points
        for id in range(self.yint.shape[0]-1): # 200 rows
            phiReal_temp, phiImag_temp = self.net_phi(u*(1.0 + yint[id+1,0]), t)
            
            # tile BinaryMat first
            Multiplier = tf.tile(BinaryMat[:,id+1:id+2],[self.Nf_t ,1])
            # make adjustments for those evaluated outside the domain
            phiReal_temp = tf.multiply(phiReal_temp, Multiplier)
            phiImag_temp = tf.multiply(phiImag_temp, Multiplier)
            
            phiReal_integ = tf.add(phiReal_integ, phiReal_temp)
            phiImag_integ = tf.add(phiImag_integ, phiImag_temp)
        
        phiReal_integ = phiReal_integ/self.nInteg
        phiImag_integ = phiImag_integ/self.nInteg
        
        f_Real = phiReal_t - self.rho*u*phiReal_u + u*phiImag_uu - self.lambdaParam*(phiReal_integ - phiReal)
        f_Imag = phiImag_t - self.rho*u*phiImag_u - u*phiReal_uu - self.lambdaParam*(phiImag_integ - phiImag)
        
        return f_Real, f_Imag
    
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self, nIter):
        
        tf_dict = {self.u0_tf: self.u0, self.t0_tf: self.t0, self.phi0Real_tf: self.phi0Real, self.phi0Imag_tf: self.phi0Imag,
                   self.uOrigin_tf: self.uOrigin, self.tOrigin_tf: self.tOrigin, self.phiOrigReal_tf: self.phiOrigReal, self.phiOrigImag_tf: self.phiOrigImag,
                   self.uf_tf: self.uf, self.tf_tf: self.tf,
                   self.yint_tf: self.yint, self.BinaryMat_tf: self.BinaryMat}
              
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
        
        # X_star points (u,t) at which the solution is evaluated
        
        tf_dict = {self.u0_tf: X_star[:,0:1],  self.t0_tf: X_star[:,1:2]}
        
        phi_star_real = self.sess.run(self.phi0_pred_Real, tf_dict)        
        phi_star_imag = self.sess.run(self.phi0_pred_Imag, tf_dict)
        
        return phi_star_real, phi_star_imag
 
    
if __name__ == "__main__": 

    noise = 0.0        
    
    # declare parameters
    lambdaParam = 12.
    rho = 2.
    a = 0.5
    unifLB = 0.5;
    unifUB = 5.5;
    
    # Domain bounds
    lb = np.array([ 0.0,  0.0]) # u, t bounds
    ub = np.array([ 10.0, 1.0])

    # Collocation points parameters
    N0 = 400
    Norigin = 100

    Nf_u = 400 # satisfying the governing equations
    Nf_t = 50 # satisfying governing equations
    N_integ = 7 # compute the integral
    
    layers = [2, 100, 100, 100, 100, 2]
    
    # for prediction
    tPred = np.array([0., 0.125, 0.25, 0.5, 0.75, 1.])
    tPred = tPred[:,None] 
    uPred = np.linspace(lb[0],ub[0],401)
    uPred = uPred[:,None]
    
    # training data
    
    # collocation points for initial condition
    uInit = np.linspace(lb[0],ub[0],N0+1)
    uInit = uInit[1:]
    U0 = np.hstack((uInit.flatten()[:,None],np.zeros((N0,1))))
    InitCondReal = (np.sin(uInit*unifUB) - np.sin(uInit*unifLB))/(uInit*(unifUB-unifLB))
    InitCondImag = (np.cos(uInit*unifLB) - np.cos(uInit*unifUB))/(uInit*(unifUB-unifLB))
    InitCond = np.hstack((InitCondReal.flatten()[:,None],InitCondImag.flatten()[:,None]))
    
    # collocation points to enforce origin condition
    tOrigin = np.linspace(lb[1],ub[1],Norigin)
    UOrigin = np.hstack((np.zeros((Norigin,1)),tOrigin.flatten()[:,None]))
    OriginCond = np.hstack((np.ones((Norigin,1)),np.zeros((Norigin,1))))

    # collocation points to enforce PDE    
    
    Uf_u = np.linspace(lb[0],ub[0],Nf_u)
    Uf_u = Uf_u[:,None]
    Uf_t = np.linspace(lb[1],ub[1],Nf_t)
    
    Uf_u_mesh, Uf_t_mesh = np.meshgrid(Uf_u, Uf_t)
    Uf = np.hstack((Uf_u_mesh.flatten()[:,None], Uf_t_mesh.flatten()[:,None]))
    
    # collocation points to compute the integral
    
    Y_int = np.linspace(-a,a,N_integ)
    Y_int = Y_int[:,None]
    
    # create BinaryMat matrix
   
    IntegrandEval = np.matmul(Uf_u,1.0 + np.transpose(Y_int))
    BinaryMat = np.ones((IntegrandEval.shape)) 
    BinaryMat[IntegrandEval > ub[0]] = 0.0
    
    # create model
    model = PhysicsInformedNN(U0, InitCond, UOrigin, OriginCond, Uf, Y_int, BinaryMat, Nf_u, Nf_t, layers, lb, ub, lambdaParam, rho, a)
            
    start_time = time.time()                
    model.train(25000)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    # Compute predicted characteristic function
   
    phiPred_Real = np.zeros((uPred.shape[0], tPred.shape[0]))
    phiPred_Imag = np.zeros((uPred.shape[0], tPred.shape[0]))
    
    for timeID in range(tPred.shape[0]):
        # form X_star to include t
        tSlice = tPred[timeID]*np.ones((uPred.shape[0],1))
        X_star = np.hstack((uPred, tSlice))
        
        # perform prediction
        phi_pred_Real, phi_pred_Imag = model.predict(X_star)
        
        phiPred_Real[:,timeID:timeID+1] = phi_pred_Real
        phiPred_Imag[:,timeID:timeID+1] = phi_pred_Imag
        
        
    #%%
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    

    plt.rc('text',usetex=False) 
    
   # plot the real part of characteristic function
    
    fig = plt.figure(figsize=(10,15))

    plt.subplot(3,2,1)
    
    plt.plot(uPred,phiPred_Real[:,0], 'b-', linewidth = 2)       
    plt.xlabel('u')
    plt.ylabel('Re[phi(u,x)]')    
    plt.title('t = %.2f' % (tPred[0]), fontsize = 10)
    
    plt.subplot(3,2,2)
    
    plt.plot(uPred,phiPred_Real[:,1], 'b-', linewidth = 2)       
    plt.xlabel('u')
    plt.ylabel('Re[phi(u,x)]')    
    plt.title('t = %.2f' % (tPred[1]), fontsize = 10)
    
    plt.subplot(3,2,3)
    
    plt.plot(uPred,phiPred_Real[:,2], 'b-', linewidth = 2)       
    plt.xlabel('u')
    plt.ylabel('Re[phi(u,x)]')    
    plt.title('t = %.2f' % (tPred[2]), fontsize = 10)
    
    plt.subplot(3,2,4)
    
    plt.plot(uPred,phiPred_Real[:,3], 'b-', linewidth = 2)       
    plt.xlabel('u')
    plt.ylabel('Re[phi(u,x)]')    
    plt.title('t = %.2f' % (tPred[3]), fontsize = 10)
    
    plt.subplot(3,2,5)
    
    plt.plot(uPred,phiPred_Real[:,4], 'b-', linewidth = 2)       
    plt.xlabel('u')
    plt.ylabel('Re[phi(u,x)]')    
    plt.title('t = %.2f' % (tPred[4]), fontsize = 10)
    
    plt.subplot(3,2,6)
    
    plt.plot(uPred,phiPred_Real[:,5], 'b-', linewidth = 2)       
    plt.xlabel('u')
    plt.ylabel('Re[phi(u,x)]')    
    plt.title('t = %.2f' % (tPred[5]), fontsize = 10)
    
    plt.show()
    
    # plot the imaginary part of characteristic function
    
    fig = plt.figure(figsize=(10,15))

    plt.subplot(3,2,1)
    
    plt.plot(uPred,phiPred_Imag[:,0], 'b-', linewidth = 2)       
    plt.xlabel('u')
    plt.ylabel('Im[phi(u,x)]')    
    plt.title('t = %.2f' % (tPred[0]), fontsize = 10)
    
    plt.subplot(3,2,2)
    
    plt.plot(uPred,phiPred_Imag[:,1], 'b-', linewidth = 2)       
    plt.xlabel('u')
    plt.ylabel('Im[phi(u,x)]')    
    plt.title('t = %.2f' % (tPred[1]), fontsize = 10)
    
    plt.subplot(3,2,3)
    
    plt.plot(uPred,phiPred_Imag[:,2], 'b-', linewidth = 2)       
    plt.xlabel('u')
    plt.ylabel('Im[phi(u,x)]')    
    plt.title('t = %.2f' % (tPred[2]), fontsize = 10)
    
    plt.subplot(3,2,4)
    
    plt.plot(uPred,phiPred_Imag[:,3], 'b-', linewidth = 2)       
    plt.xlabel('u')
    plt.ylabel('Im[phi(u,x)]')    
    plt.title('t = %.2f' % (tPred[3]), fontsize = 10)
    
    plt.subplot(3,2,5)
    
    plt.plot(uPred,phiPred_Imag[:,4], 'b-', linewidth = 2)       
    plt.xlabel('u')
    plt.ylabel('Im[phi(u,x)]')    
    plt.title('t = %.2f' % (tPred[4]), fontsize = 10)
    
    plt.subplot(3,2,6)
    
    plt.plot(uPred,phiPred_Imag[:,5], 'b-', linewidth = 2)       
    plt.xlabel('u')
    plt.ylabel('Im[phi(u,x)]')    
    plt.title('t = %.2f' % (tPred[5]), fontsize = 10)
    
    plt.show()
