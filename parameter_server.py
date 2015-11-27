
import numpy as np

class Parameter_Server:
    """
    Storing all the parameters for the network.
    Also I include several optimization methods here.
    Now this code supports:
    plain SGD, adaGrad, RMSprop, adaDelta, ADAM;
    also it supports adding momentum terms.
    """
    def __init__(self, opt_method = 'SGD', momentum = False):
        """
        Initialise the storage of parameters
        """
        self.opt_method = opt_method
        self.use_momentum = momentum
        if self.use_momentum is True:
            self.momentum = dict()
        if self.opt_method in ['ADAM', 'ADADELTA']:
            self.gradMean = dict()
        if self.opt_method != 'SGD':
            self.gradVar = dict()
        
    def init_gradient_storage(self, params):
        """
        Initialise the space to store gradient info.
        Here we assume that all parameters have been initialised.
        """
        self.t = 0     
        
        for key in params:
            if self.opt_method != 'SGD':
                self.gradVar[key] = params[key] * 0.0
            if self.opt_method in ['ADAM', 'ADADELTA']:
                self.gradMean[key] = params[key] * 0.0
            if self.use_momentum is True:
                self.momentum[key] = params[key] * 0.0
                
        # set the parameters as recommended
        if self.opt_method == 'ADAGRAD':
            self.epsi = 10e-6
        if self.opt_method in ['RMSPROP', 'ADADELTA']:
            self.betaMean = 0.9
            self.betaVar = 0.9
            self.epsi = 10e-4
        if self.opt_method == 'ADAM':
            self.betaMean = 0.9
            self.betaVar = 0.99
            self.epsi = 10e-8
        
    def update(self, params, gradients, learning_rate, momentum_rate = None):
        """
        Update the network parameters given the gradients.
        alpha is the momentum (damping) rate, default is None.
        """
        self.t += 1
        
        if self.use_momentum is True and self.opt_method != 'ADAM' \
            and momentum_rate is None:
            momentum_rate = 0.5
            
        for key in params:
            # first correct gradients with momentum if needed
            if self.use_momentum is True and self.opt_method != 'ADAM':
                if self.t == 1:
                    self.momentum[key] = gradients[key]
                gradients[key] += momentum_rate * \
                    (self.momentum[key] - gradients[key]) 
                self.momentum[key] = gradients[key]
                
            # then do optimization!
            if self.opt_method == 'SGD':
                #scale = np.linalg.norm(gradients[key])
                params[key] -= learning_rate * gradients[key] #/ scale
                    
                                                
            if self.opt_method == 'ADAGRAD':
                self.gradVar[key] += gradients[key] ** 2
                params[key] -= learning_rate / \
                    (np.sqrt(self.gradVar[key]) + self.epsi) * gradients[key]
                
            if self.opt_method == 'RMSPROP':
                self.gradVar[key] += (1 - self.betaVar) * \
                                     (gradients[key] ** 2 - self.gradVar[key])
                params[key] -= learning_rate / \
                    np.sqrt(self.gradVar[key] + self.epsi) * gradients[key]
                                     
            if self.opt_method == 'ADADELTA':
                self.gradVar[key] += (1 - self.betaVar) * \
                                     (gradients[key] ** 2 - self.gradVar[key])       
                update = gradients[key] * \
                    np.sqrt(self.gradMean[key] + self.epsi) / \
                    np.sqrt(self.gradVar[key] + self.epsi)
                # here gradMean in fact stores the difference between updates
                self.gradMean[key] += (1 - self.betaMean) * \
                                      (update ** 2 - self.gradMean[key])
                params[key] -= update               
                
            if self.opt_method == 'ADAM':
                self.gradMean[key] += (1 - self.betaMean) * \
                                      (gradients[key] - self.gradMean[key])
                self.gradVar[key] += (1 - self.betaVar) * \
                                     (gradients[key] ** 2 - self.gradVar[key])            
                alpha = learning_rate * np.sqrt(1 - self.betaVar ** self.t) \
                        / (1 - self.betaMean ** self.t)
                epsi = self.epsi * np.sqrt(1 - self.betaVar ** self.t)
                params[key] -= alpha * self.gradMean[key] \
                               / (np.sqrt(self.gradVar[key]) + epsi)
        
        return params
            
