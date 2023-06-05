import tensorflow as tf
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import warnings

DTYPE='float32'

# Define model architecture
class PINN_NeuralNet(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self, lb, ub, 
            output_dim=1,
            num_hidden_layers=8, 
            num_neurons_per_layer=20,
            activation='tanh',
            kernel_initializer='glorot_normal',
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lb = lb
        self.ub = ub
        
        # Define NN architecture
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)
        
    def call(self, X):
        """Forward-pass through neural network."""
        Z = self.scale(X)
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)
    

class PINN_PDESolver():
    def __init__(self, model, X_r, fun_r, get_r):
        self.model = model
        
        # Store collocation points
        self.t = X_r[:,0:1]
        self.x = X_r[:,1:2]
        
        # Initialize history of losses and global iteration counter
        self.hist = []
        self.iter = 0
        self.fun_r = fun_r
        self.get_r_fn = get_r
    
    def get_r(self):        
        return self.get_r_fn(t=self.t, x=self.x, model=self.model, fun_r=self.fun_r)
    
    def loss_fn(self, X, u):
        
        # Compute phi_r
        r = self.get_r()
        phi_r = tf.reduce_mean(tf.square(r))
        
        # Initialize loss
        loss = phi_r

        # Add phi_0 and phi_b to the loss
        phi_0_b = 0
        for i in range(len(X)):
            u_pred = self.model(X[i])
            phi_0_b += tf.reduce_mean(tf.square(u[i] - u_pred))
                
        loss += phi_0_b
        
        return loss, phi_r
    
    def get_grad(self, X, u):
        with tf.GradientTape(persistent=True) as tape:
            # This tape is for derivatives with
            # respect to trainable variables
            tape.watch(self.model.trainable_variables)
            loss, phi_r = self.loss_fn(X, u)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        
        return loss, phi_r, g
    
    def solve_with_TFoptimizer(self, optimizer, X, u, 
                               N=None, min_loss=None, timeout=None):
        """This method performs a gradient descent type optimization.
        Takes one stopping criterion: N number of iterations, 
        min_loss amount of minimal loss, or timeout amount of time.
        The parameter modify_lr uses and adaptive modified learning rate;
        a combination of exponential decay and piecewise constant function lr."""
        
        @tf.function
        def train_step():
            loss, phi_r, grad_theta = self.get_grad(X, u)
            
            # Perform gradient descent step, update the weights of the model.
            optimizer.apply_gradients(zip(grad_theta, self.model.trainable_variables))
            return loss, phi_r

        def loss_callback():
            loss, phi_r = train_step()
                
            self.current_loss = loss.numpy()
            self.phi_r = phi_r.numpy().astype(np.float64)
            self.phi_0_b = self.current_loss - self.phi_r
            self.callback()

        if N is not None:
            while True:        
                loss_callback()
                
                if(self.iter == N):
                    print ('{} iterations reached.'.format(self.iter))
                    break
        
        elif min_loss is not None:
            while True:
                loss_callback()
                
                if self.current_loss < min_loss:
                    print ('Loss is reached. Current loss: {:10.8e}'.format(self.current_loss))
                    break                
            
        elif timeout is not None:
            t0 = time.time()
            stop_after = time.time() + timeout
            
            while True:                
                loss_callback()
                
                if time.time() > stop_after:
                    print ('Timeout is reached. Time elapsed: {} seconds'.format(time.time()-t0))
                    break
            

    def solve_with_ScipyOptimizer(self, X, u, method='L-BFGS-B', min_loss=None, timeout=None, **kwargs):
        """This method provides an interface to solve the learning problem
        using a routine from scipy.optimize.minimize.
        (Tensorflow 1.xx had an interface implemented, which is not longer
        supported in Tensorflow 2.xx.)
        Type conversion is necessary since scipy-routines are written in Fortran
        which requires 64-bit floats instead of 32-bit floats."""
        
        def get_weight_tensor():
            """Function to return current variables of the model
            as 1d tensor as well as corresponding shapes as lists."""
            
            weight_list = []
            shape_list = []
            
            # Loop over all variables, i.e. weight matrices, bias vectors and unknown parameters
            for v in self.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
                
            weight_list = tf.convert_to_tensor(weight_list)
            return weight_list, shape_list

        x0, shape_list = get_weight_tensor()
        
        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in self.model.variables:
                vs = v.shape
                
                # Weight matrices
                if len(vs) == 2:  
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw],(vs[0],vs[1]))
                    idx += sw
                
                # Bias vectors
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx += vs[0]
                    
                # Variables (in case of parameter identification setting)
                elif len(vs) == 0:
                    new_val = weight_list[idx]
                    idx += 1
                    
                # Assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(new_val, DTYPE))
        
        def get_loss_and_grad(w):
            """Function that provides current loss and gradient
            w.r.t the trainable variables as vector. This is mandatory
            for the LBFGS minimizer from scipy."""
            
            # Update weights in model
            set_weight_tensor(w)
            # Determine value of \phi and gradient w.r.t. \theta at w
            loss, phi_r, grad = self.get_grad(X, u)
            
            # Store current loss for callback function            
            loss = loss.numpy().astype(np.float64)        
            self.current_loss = loss
            self.phi_r = phi_r.numpy().astype(np.float64)
            self.phi_0_b = self.phi_0_b = self.current_loss - self.phi_r                        
            
            # Flatten gradient
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            # Gradient list to array
            grad_flat = np.array(grad_flat,dtype=np.float64)
            
            # Return value and gradient of \phi as tuple
            return loss, grad_flat
        
        if min_loss is not None:            
            
            self.current_loss = None
            class TimedFun:
                """Class to stop scipy minimize learning when 
                certain loss is reached (if set).
                
                Done by modifying function for every call and
                raising and catching ValueError when condition is met."""
                def __init__(self, fun, stop_after):
                    self.fun_in = fun
                    self.stop_after = stop_after

                def fun(self, x, outer_instance):
                    if outer_instance.current_loss is not None and \
                       outer_instance.current_loss < self.stop_after:
                            raise ValueError("Loss is reached.")
                    self.fun_value = self.fun_in(x)
                    self.x = x
                    return self.fun_value

            get_loss_and_grad_timed = TimedFun(fun=get_loss_and_grad, stop_after=min_loss)
            res = None
            
            try:
                res = scipy.optimize.minimize(fun=get_loss_and_grad_timed.fun,
                                              x0=x0,
                                              args=(self),
                                              jac=True,
                                              method=method,
                                              callback=self.callback,
                                              **kwargs)
            except Exception as e:
                # print("Error: " + str(e))
                print (str(e) + ' Current loss: {:10.8e}'.format(self.current_loss))
                pass

            return res
        
        
        elif timeout is not None:            
            """Class to stop scipy minimize learning when 
            certain timeout is reached (if set).
            
            Done by modifying function for every call and
            raising and catching ValueError when condition is met."""

            class TimedFun:
                def __init__(self, fun, stop_after):
                    self.fun_in = fun
                    self.started = False
                    self.stop_after = stop_after

                def fun(self, x):
                    if self.started is False:
                        self.started = time.time()
                    elif abs(time.time() - self.started) >= self.stop_after:
                        raise ValueError("Timeout is reached.")
                    self.fun_value = self.fun_in(x)
                    self.x = x
                    return self.fun_value

            get_loss_and_grad_timed = TimedFun(fun=get_loss_and_grad, stop_after=timeout)
            res = None
            
            try:
                res = scipy.optimize.minimize(fun=get_loss_and_grad_timed.fun,
                                              x0=x0,
                                              jac=True,
                                              method=method,
                                              callback=self.callback,
                                              **kwargs)
            except Exception as e:
                # print("Error: " + str(e))
                print (str(e) + ' Time elapsed: {}'.format(time.time() - get_loss_and_grad_timed.started))
                pass
            
            return res
                        
            
        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method,
                                       callback=self.callback,
                                       **kwargs)
        
    def callback(self, xr=None):
        if self.iter % 50 == 0:
            print('It {:05d}: l_u = {:10.8e} l_f = {:10.8e} loss = {:10.8e}'.format(self.iter,self.phi_0_b,self.phi_r,self.current_loss))
            # print('It {:05d}: loss = {:10.8e}'.format(self.iter,self.current_loss))
        self.hist.append(self.current_loss)
        self.iter+=1        
        
    def predict_solution(self, Nt=1000, Nx=1000):
        # Set up meshgrid
        tspace = np.linspace(self.model.lb[0], self.model.ub[0], Nt+1)
        xspace = np.linspace(self.model.lb[1], self.model.ub[1], Nx+1)
        T, X = np.meshgrid(tspace, xspace)
        Xgrid = np.vstack([T.flatten(),X.flatten()]).T
        
        # Determine predictions of u(t, x)
        upred = self.model(tf.cast(Xgrid,DTYPE))
        # Reshape upred
        U = upred.numpy().reshape(Nt+1,Nx+1)

        self.T_pred, self.X_pred = T, X
        return U
    
    def plot_solution(self, U=None, **kwargs):
        # If upred has not been predicted
        if U is None:
            self.predict_solution()

        # Surface plot of solution u(t, x)
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X_pred, self.T_pred, U, cmap='viridis', **kwargs)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$t$')
        ax.set_zlabel('$u_\\theta(t,x)$')
        # ax.view_init(35,35)
        return ax
        
    def plot_loss_history(self, ax=None):
        if not ax:
            fig = plt.figure(figsize=(7,5))
            ax = fig.add_subplot(111)
        ax.semilogy(range(len(self.hist)), self.hist,'k-')
        ax.set_xlabel('$n_{epoch}$')
        ax.set_ylabel('$\\phi^{n_{epoch}}$')
        return ax  
    

class PINN_IdentificationNet(PINN_NeuralNet):
    def __init__(self, initial_lambda, *args, **kwargs):
        # Call init of base class
        super().__init__(*args,**kwargs)
        
        # Receives list of initial lambdas as input.
        # Type casting is done to cast list to numpy array.
        # Floats will not be casted.
        # try:
        #     len(initial_lambda)
        #     self.lambd = []
        #     for lambd in initial_lambda:
        #         self.lambd.append(tf.Variable(initial_value=lambd, trainable=True, dtype=DTYPE))
        # except TypeError as e:
        #     # Initialize variable for lambda
        #     self.lambd = tf.Variable(initial_value=initial_lambda, trainable=True, dtype=DTYPE)
        #     pass
        
        try:
            len(initial_lambda)
            self.initial_lambda = np.array(initial_lambda, dtype=DTYPE)
        except TypeError as e:
            self.initial_lambda = initial_lambda

        # Initialize variable for lambda
        self.lambd = tf.Variable(initial_value=self.initial_lambda, trainable=True, dtype=DTYPE)

        self.lambd_list = []

    
class PINN_IdentificationSolver(PINN_PDESolver):
    def get_r(self):
        return self.get_r_fn(t=self.t, x=self.x, model=self.model, fun_r=self.fun_r)
        
    def callback(self, xr=None):
        # Change tensorflow variable to numpy
        # try:
        #     lambd = []
        #     for i in range(len(self.model.lambd)):
        #         lambd.append(self.model.lambd[i].numpy())
        # except TypeError as e:
        #     lambd = self.model.lambd.numpy()
        
        lambd = self.model.lambd.numpy()

        self.model.lambd_list.append(lambd)
        
        if self.iter % 50 == 0:
            print('It {:05d}: loss = {:10.8e} lambda = '.format(self.iter, self.current_loss) + str(lambd))
        
        self.hist.append(self.current_loss)
        self.iter += 1
        
    def plot_loss_and_param(self, axs=None):
        if axs:
            ax1, ax2 = axs
            self.plot_loss_history(ax1)
        else:
            ax1 = self.plot_loss_history()
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.plot(range(len(self.hist)), self.model.lambd_list,'-',color=color)
        ax2.set_ylabel('$\\lambda^{n_{epoch}}$', color=color)
        return (ax1,ax2)


class Hybrid_IdentificationNet(PINN_NeuralNet):
    def __init__(self, initial_lambda, *args, **kwargs):        
        # Call init of base class
        super().__init__(*args,**kwargs)
        self.initial_lambda = initial_lambda  


    def add_param(self):
        # Add parameters to the model after 
        # fitting with given data.

        try:
            len(self.initial_lambda)
            self.initial_lambda = np.array(self.initial_lambda, dtype=DTYPE)
        except TypeError as e:
            pass

        # Initialize variable for lambda
        self.lambd = tf.Variable(initial_value=self.initial_lambda, trainable=True, dtype=DTYPE)
        self.lambd_list = []



class Hybrid_IdentificationSolver(PINN_PDESolver):
    def solve_with_TFoptimizer(self, optim_fwd, optim_param, get_r_param, X, u, 
                               N_fwd=None, min_loss_fwd=None, timeout_fwd=None, 
                               N_param=None, min_loss_param=None, timeout_param=None, modified=False, **kwargs):
        self.changed = False
        self.lambd_list = []
        super().solve_with_TFoptimizer(optim_fwd, X, u, N=N_fwd, min_loss=min_loss_fwd, timeout=timeout_fwd, **kwargs)
        
        # Add parameters after fitting 
        # with given data.
        self.model.add_param()
        self.changed = True

        print('\n\nFinding PDE parameters.')

        if modified is True:

            @tf.function
            def train_step_param():
                loss, grad_theta = get_r_param()
                
                # Perform gradient descent step, update the weights of the model.
                optim_param.apply_gradients(zip([grad_theta], [self.model.lambd]))
                return loss

            def loss_callback():
                loss = train_step_param()

                self.current_loss = loss.numpy()
                callback()
        
        else:
            @tf.function
            def loss_fn_param():                
                # phi_r is calculated only after
                # variables are added.
                # Compute phi_r
                r = self.get_r()
                phi_r = tf.reduce_mean(tf.square(r))
                
                loss = phi_r

                return loss

            def loss_callback():
                optim_param.minimize(loss_fn_param, [self.model.lambd])
                callback()
        

        def callback():
                lambd = self.model.lambd.numpy()
                self.model.lambd_list.append(lambd)
                
                if self.iter % 50 == 0:
                    print('It {:05d}: loss = {:10.8e} lambda = '.format(self.iter, self.current_loss) + str(lambd))

                self.hist.append(self.current_loss)
                self.iter += 1

        if N_param is not None:
            for _ in range(N_param):
                loss_callback()
            
            print ('{} iterations reached.'.format(N_param))
        
        elif min_loss_param is not None:
            while True:
                loss_callback()
                
                if self.current_loss < min_loss_param:
                    print ('Loss is reached. Current loss: {:10.8e}'.format(self.current_loss))
                    break                
            
        elif timeout_param is not None:
            t0 = time.time()
            stop_after = time.time() + timeout_param
            
            while True:                
                loss_callback()

                if time.time() > stop_after:
                    print ('Timeout is reached. Time elapsed: {} seconds'.format(time.time()-t0))
                    break
            
    def solve_with_ScipyOptimizer(self, X, u, method_fwd='L-BFGS-B', method_param='L-BFGS-B', 
                                  min_loss_fwd=None, timeout_fwd=None, 
                                  min_loss_param=None, timeout_param=None, **kwargs):
        """This method provides an interface to solve the learning problem
        using a routine from scipy.optimize.minimize.
        (Tensorflow 1.xx had an interface implemented, which is not longer
        supported in Tensorflow 2.xx.)
        Type conversion is necessary since scipy-routines are written in Fortran
        which requires 64-bit floats instead of 32-bit floats."""
        self.changed = False
        self.lambd_list = []
        super().solve_with_ScipyOptimizer(X, u, method_fwd, min_loss=min_loss_fwd, timeout=timeout_fwd, **kwargs)
        
        # Add parameters after fitting 
        # with given data.
        self.model.add_param()
        self.changed = True

        print('\n\nFinding PDE parameters.')
        
        x0 = tf.convert_to_tensor([self.model.lambd])
        
        def set_weight_tensor(weight_list):
            """Function which sets list of weights
            to variables in the model."""
            idx = 0
            for v in self.model.lambd:
                new_val = weight_list[idx]
                idx += 1
                    
                # Assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(new_val, DTYPE))
        
        def get_loss_and_grad(w):
            """Function that provides current loss and gradient
            w.r.t the trainable variables as vector. This is mandatory
            for the LBFGS minimizer from scipy."""
            
            # Update weights in model
            set_weight_tensor(w)
            # Determine value of \phi and gradient w.r.t. \theta at w
            loss, phi_r, grad = self.get_grad(X, u)
            
            # Store current loss for callback function            
            loss = loss.numpy().astype(np.float64)        
            self.current_loss = loss
            self.phi_r = phi_r.numpy().astype(np.float64)
            self.phi_0_b = self.phi_0_b = self.current_loss - self.phi_r                        
            
            # Flatten gradient
            grad_flat = []
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            # Gradient list to array
            grad_flat = np.array(grad_flat,dtype=np.float64)
            
            # Return value and gradient of \phi as tuple
            return loss, grad_flat
        
        if min_loss_param is not None:            
            
            self.current_loss = None
            class TimedFun:
                """Class to stop scipy minimize learning when 
                certain loss is reached (if set).
                
                Done by modifying function for every call and
                raising and catching ValueError when condition is met."""
                def __init__(self, fun, stop_after):
                    self.fun_in = fun
                    self.stop_after = stop_after

                def fun(self, x, outer_instance):
                    if outer_instance.current_loss is not None and \
                       outer_instance.current_loss < self.stop_after:
                            raise ValueError("Loss is reached.")
                    self.fun_value = self.fun_in(x)
                    self.x = x
                    return self.fun_value

            get_loss_and_grad_timed = TimedFun(fun=get_loss_and_grad, stop_after=min_loss_param)
            res = None
            
            try:
                res = scipy.optimize.minimize(fun=get_loss_and_grad_timed.fun,
                                              x0=x0,
                                              args=(self),
                                              jac=True,
                                              method=method_param,
                                              callback=self.callback,
                                              **kwargs)
            except Exception as e:
                # print("Error: " + str(e))
                print (str(e) + ' Current loss: {:10.8e}'.format(self.current_loss))
                pass

            return res
        
        
        elif timeout_param is not None:            
            """Class to stop scipy minimize learning when 
            certain timeout is reached (if set).
            
            Done by modifying function for every call and
            raising and catching ValueError when condition is met."""

            class TimedFun:
                def __init__(self, fun, stop_after):
                    self.fun_in = fun
                    self.started = False
                    self.stop_after = stop_after

                def fun(self, x):
                    if self.started is False:
                        self.started = time.time()
                    elif abs(time.time() - self.started) >= self.stop_after:
                        raise ValueError("Timeout is reached.")
                    self.fun_value = self.fun_in(x)
                    self.x = x
                    return self.fun_value

            get_loss_and_grad_timed = TimedFun(fun=get_loss_and_grad, stop_after=timeout_param)
            res = None
            
            try:
                res = scipy.optimize.minimize(fun=get_loss_and_grad_timed.fun,
                                              x0=x0,
                                              jac=True,
                                              method=method_param,
                                              callback=self.callback,
                                              **kwargs)
            except Exception as e:
                # print("Error: " + str(e))
                print (str(e) + ' Time elapsed: {}'.format(time.time() - get_loss_and_grad_timed.started))
                pass
            
            return res
                        
            
        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                       x0=x0,
                                       jac=True,
                                       method=method_param,
                                       callback=self.callback,
                                       **kwargs)
    
    def loss_fn(self, X, u):
        # Initialize loss
        loss = 0
        phi_r = 0

        # phi_r is calculated only after
        # variables are added.
        if self.changed is not False:
            # Compute phi_r
            r = self.get_r()
            phi_r = tf.reduce_mean(tf.square(r))
            
            loss += phi_r

        else:
            # Add phi_0 and phi_b to the loss
            for i in range(len(X)):
                u_pred = self.model(X[i])
                loss += tf.reduce_mean(tf.square(u[i] - u_pred))
        
        return loss, phi_r

    def callback(self, xr=None):
        if self.iter % 50 == 0:
            print('It {:05d}: l_u = {:10.8e} loss = {:10.8e}'.format(self.iter,self.phi_0_b,self.current_loss))
            # print('It {:05d}: loss = {:10.8e}'.format(self.iter,self.current_loss))
        self.hist.append(self.current_loss)
        self.iter+=1



def generate_training_data(N_0, N_b, fun_u_0, fun_u_b, lb, ub):
    # Draw uniform sample points for initial boundary data
    t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0]
    x_0 = tf.random.uniform((N_0,1), lb[1], ub[1], dtype=DTYPE)
    X_0 = tf.concat([t_0, x_0], axis=1)

    # Evaluate initial condition at x_0
    u_0 = fun_u_0(x_0)

    # Boundary data
    t_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
    x_b = lb[1] + (ub[1] - lb[1]) * tf.keras.backend.random_bernoulli((N_b,1), 0.5, dtype=DTYPE)
    X_b = tf.concat([t_b, x_b], axis=1)

    # Evaluate boundary condition at (t_b,x_b)
    u_b = fun_u_b(t_b, x_b)    

    # Collect boundary and inital data in lists
    X_data = [X_0, X_b]
    u_data = [u_0, u_b]

    return X_data, u_data

def generate_collocation_points(N_r, lb, ub):
    # Draw uniformly sampled collocation points
    t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
    x_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
    X_r = tf.concat([t_r, x_r], axis=1)
    
    return X_r

def generate_training_data_inverse(N_d, lb, ub, u_expl, noise=0):
    # Draw points with measurements randomly
    t_d = tf.random.uniform((N_d,1), lb[0], ub[0], dtype=DTYPE)
    x_d = tf.random.uniform((N_d,1), lb[1], ub[1], dtype=DTYPE)
    X_d = tf.concat([t_d, x_d], axis=1)
    
    # Explicit analytical solution
    u_d = u_expl(t_d, x_d)
    u_d += noise * tf.random.normal(u_d.shape, dtype=DTYPE)
    
    return X_d, u_d

def predict_solution(model, Nt=1000, Nx=1000):
        # Set up meshgrid
        tspace = np.linspace(model.lb[0], model.ub[0], Nt+1)
        xspace = np.linspace(model.lb[1], model.ub[1], Nx+1)
        T, X = np.meshgrid(tspace, xspace)
        Xgrid = np.vstack([T.flatten(),X.flatten()]).T
        
        # Determine predictions of u(t, x)
        upred = model(tf.cast(Xgrid,DTYPE))
        # Reshape upred
        U = upred.numpy().reshape(Nt+1,Nx+1)
        return U

def evaluate_solution(u_pred, u_star):
    return np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2) * 100

def evaluate_solution_log_time(u_pred, u_star):
    return np.log(np.sum(np.abs(u_star-u_pred), axis=0)/u_star.shape[0])

def evaluate_param(lambda_pred, lambda_star):
    if type(lambda_pred) == list:
        # If the input is a list of predictions,
        # the average of the predictions will 
        # be used in the evaluation.
        lambda_pred = sum(lambda_pred)/len(lambda_pred)

    return abs(lambda_star-lambda_pred) * 100

def plot_solution(u_pred, lb, ub, time):
    idx = time/(lb[0] - ub[0]) * u_pred.shape[1]
    plt.plot(u_pred[:, idx])
