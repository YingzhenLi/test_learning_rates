import autograd.numpy as np
from functions import make_functions, init_params
from parameter_server import Parameter_Server
from autograd import value_and_grad
from autograd.util import quick_grad_check, check_grads
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from matplotlib.colors import LogNorm
import argparse

"plot the trajectory of optimization"

def inject_noise(gradients, noise_level = 0.0):
    # add Gaussian noise to the gradients
    if noise_level <= 0.0:
        return gradients
    for key in gradients:
        gradients[key] += np.random.randn() * noise_level
    return gradients

def opt_traj(func, fdict, T, opt_method = 'SGD', init = None, \
    learning_rate = 0.1, seed = 100, momentum = False, noise_level = 0.0):
    # do optimization and return the trajectory
    params = {'x': 0.0, 'y': 0.0}
    domain = fdict['domain']
    optimum = fdict['optimum']
    loss_and_grad = value_and_grad(func)
    #quick_grad_check(func, params)   
    params = init_params(params, domain, init, seed)
    check_grads(func, params)
    opt_server = Parameter_Server(opt_method, momentum)
    opt_server.init_gradient_storage(params)
    
    x_traj = []
    y_traj = []
    f_traj = []
    
    print 'optimising function using %s...' % opt_method
    for t in xrange(T):
        (func_value, func_grad) = loss_and_grad(params)
        x_traj.append(params['x'])
        y_traj.append(params['y'])
        f_traj.append(func_value)
        func_grad = inject_noise(func_grad, noise_level)
        if opt_method == 'SGD':
            norm = np.sqrt(func_grad['x'] ** 2 + func_grad['y'] ** 2)
            if norm >= 2.0:
                func_grad['x'] /= norm / 2; func_grad['y'] /= norm / 2
        params = opt_server.update(params, func_grad, learning_rate)

    return np.array(x_traj), np.array(y_traj), np.array(f_traj)

def show_anim(T, results, fdict, **kwargs):

    fig = plt.figure(figsize=(6, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax = [ax1, ax2]
    #fig, ax = plt.subplots(2)
    func = fdict['func']
    domain = fdict['domain']
    optimum = fdict['optimum']
    
    # lines in contour plots
    linewidth = 5
    line1, = ax[0].plot([], [], linewidth=linewidth, color='r', label='SGD')
    line2, = ax[0].plot([], [], linewidth=linewidth, color='c', label='ADAGRAD')
    line3, = ax[0].plot([], [], linewidth=linewidth, color='g', label='RMSPROP')
    line4, = ax[0].plot([], [], linewidth=linewidth, color='m', label='ADADELTA')
    line5, = ax[0].plot([], [], linewidth=linewidth, color='y', label='ADAM')
    
    # lines in error plots
    err1, = ax[1].plot([], [], linewidth=linewidth, color='r', label='SGD')
    err2, = ax[1].plot([], [], linewidth=linewidth, color='c', label='ADAGRAD')
    err3, = ax[1].plot([], [], linewidth=linewidth, color='g', label='RMSPROP')
    err4, = ax[1].plot([], [], linewidth=linewidth, color='m', label='ADADELTA')
    err5, = ax[1].plot([], [], linewidth=linewidth, color='y', label='ADAM')
    
    # compute error
    opt_value = func(optimum[0], optimum[1])
    max_err = 0.0
    for opt in results.keys():
        results[opt][2] = np.sqrt((results[opt][2] - opt_value)**2)
        if max_err < results[opt][2].max():
            max_err = results[opt][2].max()
    
    def init():
        offset = 2.0
        #if optimum[0] < np.inf:
        #    xmin = min(results['ADAM'][0][0], optimum[0]) - offset
        #    xmax = max(results['ADAM'][0][0], optimum[0]) + offset
        #else:
        xmin = domain[0, 0]
        xmax = domain[0, 1]
        #if optimum[1] < np.inf:
        #    ymin = min(results['ADAM'][1][0], optimum[1]) - offset
        #    ymax = max(results['ADAM'][1][0], optimum[1]) + offset
        #else:
        ymin = domain[1, 0]
        ymax = domain[1, 1]
        x = np.arange(xmin, xmax, 0.01)
        y = np.arange(ymin, ymax, 0.01)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros(np.shape(Y))
        for a, _ in np.ndenumerate(Y):
            Z[a] = func(X[a], Y[a])
        level = fdict['level']
        if level is None:
            level = np.linspace(Z.min(), Z.max(), 20)
        else:
            if level[0] == 'normal':
                level = np.linspace(Z.min(), Z.max(), level[1])
            if level[0] == 'log':
                level = np.logspace(np.log(Z.min()), np.log(Z.max()), level[1])
        CF = ax[0].contour(X,Y,Z, levels=level)
        #plt.colorbar(CF, orientation='horizontal', format='%.2f')
        ax[0].grid()
        ax[0].plot(results['ADAM'][0][0], results['ADAM'][1][0], 
            'h', markersize=15, color = '0.75')
        if optimum[0] < np.inf and optimum[1] < np.inf:
            ax[0].plot(optimum[0], optimum[1], '*', markersize=40, 
                markeredgewidth = 2, alpha = 0.5, color = '0.75')
        ax[0].legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15))
        
        ax[1].plot(0, results['ADAM'][2][0], 'o')
        ax[1].axis([0, T, -0.5, max_err + 0.5])
        ax[1].set_xlabel('num. iteration')
        ax[1].set_ylabel('loss')
        
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        line5.set_data([], [])
        
        err1.set_data([], [])
        err2.set_data([], [])
        err3.set_data([], [])
        err4.set_data([], [])
        err5.set_data([], [])
        
        return line1, line2, line3, line4, line5, \
            err1, err2, err3, err4, err5, 
            
        
    # lines
    markersize = 15
    circle1, = ax[0].plot([], [], 'ro', markersize = markersize)
    circle2, = ax[0].plot([], [], 'co', markersize = markersize)
    circle3, = ax[0].plot([], [], 'go', markersize = markersize)
    circle4, = ax[0].plot([], [], 'mo', markersize = markersize)
    circle5, = ax[0].plot([], [], 'yo', markersize = markersize)
    num_iter = np.arange(T)

    # Function to draw the actively changing part of the plot
    def animate(i):
        line1.set_data(results['SGD'][0][:i], results['SGD'][1][:i])
        line2.set_data(results['ADAGRAD'][0][:i], results['ADAGRAD'][1][:i])
        line3.set_data(results['RMSPROP'][0][:i], results['RMSPROP'][1][:i])
        line4.set_data(results['ADADELTA'][0][:i], results['ADADELTA'][1][:i])
        line5.set_data(results['ADAM'][0][:i], results['ADAM'][1][:i])
        
        circle1.set_data(results['SGD'][0][i], results['SGD'][1][i])
        circle2.set_data(results['ADAGRAD'][0][i], results['ADAGRAD'][1][i])
        circle3.set_data(results['RMSPROP'][0][i], results['RMSPROP'][1][i])
        circle4.set_data(results['ADADELTA'][0][i], results['ADADELTA'][1][i])
        circle5.set_data(results['ADAM'][0][i], results['ADAM'][1][i])
        
        err1.set_data(num_iter[:i], results['SGD'][2][:i])
        err2.set_data(num_iter[:i], results['ADAGRAD'][2][:i])
        err3.set_data(num_iter[:i], results['RMSPROP'][2][:i])
        err4.set_data(num_iter[:i], results['ADADELTA'][2][:i])
        err5.set_data(num_iter[:i], results['ADAM'][2][:i])
        
        return line1, line2, line3, line4, line5, \
            circle1, circle2, circle3, circle4, circle5, \
            err1, err2, err3, err4, err5, 

    # Create animaiton
    interval = kwargs['interval']
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=T, interval=interval, blit=True)
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='comparing learning rates.')
    parser.add_argument('--function', '-f', type=str, default='sphere')
    parser.add_argument('--iter', '-t', type=int, default=100)
    parser.add_argument('--learning_rate', '-l', type=float, default=0.1)
    parser.add_argument('-x', '-x', type=float, default=1.0)
    parser.add_argument('-y', '-y', type=float, default=1.2)
    parser.add_argument('--use_momentum', '-m', action='store_true', default=False)
    parser.add_argument('-noise_level', '-n', type=float, default=0.0)
    
    args = parser.parse_args()
    func_name = args.function
    T = args.iter
    lr = args.learning_rate
    init = np.array([args.x, args.y])
    if func_name == 'saddle2':
        init = np.array([-1.0, -np.sqrt(3)])
    momentum = args.use_momentum
    noise_level = args.noise_level
    
    results = {'SGD': [], 'ADAGRAD': [], 'RMSPROP': [], 
        'ADADELTA': [], 'ADAM': []}
    func, fdict = make_functions(func_name)
    print 'optimising %s function:' % func_name
    for opt in results.keys():
        x_traj, y_traj, f_traj  = opt_traj(func, fdict, T, opt_method = opt, 
            init = init, learning_rate = lr, momentum = momentum,
            noise_level = noise_level)
        results[opt] = [x_traj, y_traj, f_traj]
    print 'optimum:', fdict['optimum']
    show_anim(T, results, fdict, interval = 50)

