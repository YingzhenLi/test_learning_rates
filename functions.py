import autograd.numpy as np

"Test functions for optimization, here we only do 2D case."

def make_functions(name = 'sphere'):
    
    fdict = {'foo': {'func': foo,
                     'domain': np.array([[-5, 5], [-5, 5]]),
                     'optimum': np.array([0, 0]),
                     'level': None},
             'Ackley': {'func': Ackley, 
                        'domain': np.array([[-5, 5], [-5, 5]]),
                        'optimum': np.array([0, 0]),
                        'level': ['normal', 10]},
             'sphere': {'func': sphere, 
                        'domain': np.array([[-3, 3], [-3, 3]]),
                        'optimum': np.array([0, 0]),
                        'level': ['normal', 20]},
             'Rosenbrock': {'func': Rosenbrock, 
                            'domain': np.array([[-5, 5], [-5, 5]]),
                            'optimum': np.array([1, 1]),
                            'level': ['normal', 50]},
             'Beale': {'func': Beale, 
                       'domain': np.array([[-3.5, 3.5], [-3.5, 3.5]]),
                       'optimum': np.array([3, 0.5]),
                       'level': ['log', 80] },
             'Goldstein_Price': {'func': Goldstein_Price, 
                       'domain': np.array([[-2, 2], [-2, 2]]),
                       'optimum': np.array([0, -1]),
                       'level': ['normal', 50]},
             'Booth': {'func': Booth, 
                       'domain': np.array([[-6, 6], [-6, 6]]),
                       'optimum': np.array([1, 3]),
                       'level': ['normal', 50]},
             'Matyas': {'func': Matyas, 
                        'domain': np.array([[-6, 6], [-6, 6]]),
                        'optimum': np.array([0, 0]),
                        'level': ['normal', 50]},
             'McCormick': {'func': McCormick, 
                           'domain': np.array([[-1.5, 4], [-3, 4]]),
                           'optimum': np.array([-0.54719, -1.54719]),
                           'level': ['normal', 50]},
             'monkey': {'func': monkey, 
                        'domain': np.array([[-5, 5], [-5, 5]]),
                        'optimum': np.array([np.inf, np.inf]),
                        'level': ['normal', 50]},              
             'Styblinski_Tang': {'func': Styblinski_Tang, 
                           'domain': np.array([[-5, 5], [-5, 5]]),
                           'optimum': np.array([-2.903534, -2.903534]),
                           'level': ['normal', 50]},
             'saddle': {'func': saddle, 
                        'domain': np.array([[-5, 5], [-5, 5]]),
                        'optimum': np.array([0, -np.inf]),
                        'level': ['normal', 50]}, 
             'saddle2': {'func': saddle2, 
                         'domain': np.array([[-5, 5], [-5, 5]]),
                         'optimum': np.array([0, -np.inf]),
                         'level': ['normal', 50]}, 
             }
    
    def function(params):
        f = fdict[name]['func'](params['x'], params['y'])
        return f

    return function, fdict[name]
    
def init_params(params, domain, init = None, seed = 0):
    # initialise the location of x and y
    if init is not None:
        re_init = False
        if init[0] <= domain[0, 0] or init[0] >= domain[0, 1]:
            re_init = True
        if init[1] <= domain[1, 0] or init[1] >= domain[1, 1]:
            re_init = True
        if re_init is True:
            print 'invalid initialisation, do random init instead...'
            init = None
    
    if init is None:
        # random initialisation
        init = np.zeros(2)
        np.random.seed(seed)
        init[0] = np.random.random() - 0.5
        init[1] = np.random.random() - 0.5
        init[0] = init[0] * (domain[0, 1]-domain[0, 0]) + 0.5 * (np.sum(domain[0]))
        init[1] = init[1] * (domain[1, 1]-domain[1, 0]) + 0.5 * (np.sum(domain[1]))
              
    params['x'] = init[0]
    params['y'] = init[1]       
    print 'initialise x =', init[0], 'y =', init[1]
        
    return params

def foo(x, y):
    f = 0
    return f

def Ackley(x, y):
    f = -20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) \
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) \
        + np.exp(1) + 20
    return f
    
def sphere(x, y):
    f = x ** 2 + y ** 2
    return f
    
def Rosenbrock(x, y):
    f = 5 * (y - x ** 2) ** 2 + (x - 1) ** 2
    return f
    
def Beale(x, y):
    f = (1.5 - x + x * y) ** 2 \
        + (2.25 - x + x * y ** 2) ** 2 \
        + (2.625 - x + x * y ** 3) ** 2
    return f
    
def Goldstein_Price(x, y):
    f = (1 + (x + y + 1) ** 2 * \
        (19 - 14 * x + 3 * x ** 2 - 14 * y - 6 * x * y + 3 * x ** 2)) \
        * (30 + (2 * x - 3 * y) ** 2 * \
        (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))
    return f

def Booth(x, y):
    f = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
    return f
    
def Matyas(x, y):
    f = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return f
    
def McCormick(x, y):
    f = np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1
    return f
    
def monkey(x, y):
    f = x ** 3 - 3 * x * y ** 2
    return f

def Styblinski_Tang(x, y):
    f = 0.5 * (x ** 4 - 16 * x ** 2 + 5 * x) \
        + 0.5 * (y ** 4 - 16 * y ** 2 + 5 * y)
    return f
    
def saddle(x, y):
    f = x ** 2 - y ** 2
    return f
    
def saddle2(x, y):	# do (-1.0, -np.sqrt(3))
    f = (x + np.sqrt(3) * y) ** 2 - (-np.sqrt(3) * x + y) ** 2
    return 0.25 * f
 
