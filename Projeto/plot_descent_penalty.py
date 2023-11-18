import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sympy as sym

def styles_of_line():
    linestyle_tuple = [
        ('long dash with offset', (5, (10, 3))),
        ('loosely dashed',        (0, (5, 10))),
        #  ('dashed',                (0, (5, 5))),
        #  ('densely dashed',        (0, (5, 1))),

        #  ('loosely dashdotted',    (0, (3, 10, 1, 10))),
        ('dashdotted',            (0, (3, 5, 1, 5))),
        #  ('densely dashdotted',    (0, (3, 1, 1, 1))),

        ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
        #  ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        #  ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1))),
        
        ('loosely dotted',        (0, (1, 10))),
        ('dotted',                (0, (1, 1))),
        #  ('densely dotted',        (0, (1, 1))),
        ]
    
    return linestyle_tuple


def plot_3d_surface(x, f, f_obj, plot_h, plot_g, f_constraints):
    """
    Generates a 3d surface plot, showing the direction of the optimization
    
    Inputs:
        x (list): contains a list of points
        f (list): values of the points
        f_obj (method): objective function
        plot_h (boolean): True or False, for plotting the equality constraints
        plot_g (boolean): True or False, for plotting the inequality constraints
    """
    
    if isinstance(x, list):
        x = np.array(x)
    
    if len(x.shape) == 3:
        x = x[0, :, :]

    if isinstance(f, list):
        f = np.array(f)
        
    if len(f.shape) == 2:
        f = f[0, :]
        
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
    # ax = Axes3D(fig)
    
    xmin = np.floor(-np.max(np.abs(x[0, :])) - 2)
    xmax = np.ceil(np.max(np.abs(x[0, :])) + 2)
    x1 = np.arange( xmin, xmax, 0.25 )
    x2 = np.copy(x1)
    X1, X2 = np.meshgrid(x1, x2)
    
    y = np.zeros((X1.shape))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            f_grid = f_obj([X1[i,j], X2[i,j]])
            y[i,j] = f_grid[0]
    
    # To use in the constraints plot
    # ymax = np.max(y)
    # ymin = np.min(y)

    surface = ax.plot_surface(X1, X2, y,
                          cmap=cm.coolwarm,
                          rstride=1, cstride=1, alpha=0.6)
    
    
    ax.view_init(elev=20., azim=30)
    ax.set_xlabel(r'$x_1$', labelpad=20, fontsize=24, fontweight='bold')
    ax.set_ylabel(r'$x_2$', labelpad=20, fontsize=24, fontweight='bold')
    ax.set_title('Cost function')
    
    # Beginning
    ax.plot([x[0, 0]], [x[0, 1]], [f[0]] , markerfacecolor='y', markeredgecolor='y', marker='X', markersize=12)

    # Other points
    ax.plot([t[0] for t in x], [t[1] for t in x], f , markerfacecolor='k', markeredgecolor='k', marker='.', markersize=7)
    
    # Ending
    ax.plot([x[-1, 0]], [x[-1, 1]], [f[-1]] , markerfacecolor='g', markeredgecolor='g', marker='X', markersize=12)
    
    def plot_constraints(equality=True):
        linestyle_tuple = styles_of_line()
        f_grid = f_constraints(np.array([0, 0]))
        handles = []
        
        if equality:
            x2 = f_grid[0]
            cor = 'g'
            label = 'Equality constraint '
        else:
            x2 = f_grid[2]
            cor = 'y'
            label = 'Inequality constraint '
            
        n_constraints = x2.size
        
        for constraint in range(n_constraints):
            xx1 = np.array([])
            xx2 = np.array([])
            yy = np.array([])
            
            x1s = sym.Symbol('x1s')
            x2s = sym.Symbol('x2s')
            depends_on_x1, depends_on_x2 = True, True
            
            # test if it is dependent of x1
            f_grid = f_constraints(np.array([x1s, x1[0]]))
            
            if equality:
                x2 = f_grid[0]
            else:
                x2 = f_grid[2]
            
            if n_constraints > 1:
                x2 = x2[constraint]
            
            x2 = sym.solve(x2, x1s)
            
            if len(x2) == 0:
                depends_on_x1 = False
            
            # test if it is dependent of x2
            f_grid = f_constraints(np.array([x1[0], x2s]))
            
            if equality:
                x2 = f_grid[0]
            else:
                x2 = f_grid[2]
            
            if n_constraints > 1:
                x2 = x2[constraint]
            
            x2 = sym.solve(x2, x2s)
            
            if len(x2) == 0:
                depends_on_x2 = False
            
            for i in range(x1.shape[0]):
                if not depends_on_x2:
                    f_grid = f_constraints(np.array([x2s, 0]))
                else:
                    f_grid = f_constraints(np.array([x1[i], x2s]))
                
                # get the value of x2
                if equality:
                    x2 = f_grid[0]
                else:
                    x2 = f_grid[2]
                
                if n_constraints > 1:
                    x2 = x2[constraint]
                
                # for all results of x2, associate x1 and y values
                x2 = sym.solve(x2, x2s)
                x2_temp = np.array([])
                x1_temp = np.array([])
                yy_temp = np.array([])
                for j in range(len(x2)):
                    if not x2[j].is_real:
                        break
                    x2_value = float(x2[j])
                    
                    if not depends_on_x2:
                        yy_temp = np.append(yy_temp, f_obj([x2_value, x1[i]])[0])
                        x2_temp = np.append(x2_temp, x1[i])
                        x1_temp = np.append(x1_temp, x2_value)
                    else:
                        x1_temp = np.append(x1_temp, x1[i])
                        x2_temp = np.append(x2_temp, x2_value)
                        yy_temp = np.append(yy_temp, f_obj([x1[i], x2_value])[0])
                
                if not x2[j].is_real:
                    continue
                    
                if not xx1.size:
                    xx1 = np.array([]).reshape(0, x1_temp.size)
                    
                if not xx2.size:
                    xx2 = np.array([]).reshape(0, x2_temp.size)
                    
                if not yy.size:
                    yy = np.array([]).reshape(0, yy_temp.size)
                
                x1_temp = np.array([x1_temp]).T
                x2_temp = np.array([x2_temp]).T
                yy_temp = np.array([yy_temp]).T
                
                xx1 = np.hstack([xx1, x1_temp]) if xx1.size else x1_temp
                xx2 = np.hstack([xx2, x2_temp]) if xx2.size else x2_temp
                yy = np.hstack([yy, yy_temp]) if yy.size else yy_temp
            
            for row in range(xx1.shape[0]):
                h, = ax.plot(xx1[row, :], xx2[row, :], yy[row, :], linestyle=linestyle_tuple[constraint][1], color=cor, label=label+str(constraint+1))
            handles.append(h)
        ax.legend(handles=handles)

    # Plot equality constraints
    if plot_h and f_constraints([X1[0,0], X2[0,0]])[0].size > 0:
        plot_constraints(equality=True)
    
    # Plot inequality constraints
    if plot_g and f_constraints([X1[0,0], X2[0,0]])[2].size > 0:
        plot_constraints(equality=False)
    
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((xmin, xmax))
    
    plt.show()


def plot_2d_contour(x, f_obj, plot_h, plot_g, f_constraints):
    """
    Generates a contour plot, showing the direction of the optimization
    
    Inputs:
        x (list): contains a list of points
        f_obj (method): objective function
    """
    
    if isinstance(x, list):
        x = np.array(x)
    
    if len(x.shape) == 3:
        x = x[0, :, :]
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

    xmin = np.floor(-np.max(np.abs(x[0, :])) - 2)
    xmax = np.ceil(np.max(np.abs(x[0, :])) + 2)
    x1 = np.arange( xmin, xmax, 0.25 )
    x2 = np.copy(x1)
    X1, X2 = np.meshgrid(x1, x2)
    
    y = np.zeros((X1.shape))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            f_grid = f_obj([X1[i,j], X2[i,j]])
            y[i,j] = f_grid[0]

    contours = ax.contour(X1, X2, y, 30)
    ax.clabel(contours)

    colors = ['k']
    for j in range(1, len(x)):
        ax.annotate('', xy=x[j], xytext=x[j-1],
                    arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                    va='center', ha='center')
    ax.scatter(*zip(*x), c=colors, s=40, lw=0)

    # Labels, titles and a legend
    ax.set_xlabel(r'$x_1$', labelpad=20, fontsize=24, fontweight='bold')
    ax.set_ylabel(r'$x_2$', labelpad=20, fontsize=24, fontweight='bold')
    ax.set_title('Cost function')
    
    def plot_constraints(equality=True):
        linestyle_tuple = styles_of_line()
        f_grid = f_constraints(np.array([0, 0]))
        handles = []
        
        if equality:
            x2 = f_grid[0]
            cor = 'c'
            label = 'Equality constraint '
        else:
            x2 = f_grid[2]
            cor = 'm'
            label = 'Inequality constraint '
            
        n_constraints = x2.size
        
        for constraint in range(n_constraints):
            xx1 = np.array([])
            xx2 = np.array([])
            yy = np.array([])
            x1s = sym.Symbol('x1s')
            x2s = sym.Symbol('x2s')
            depends_on_x1, depends_on_x2 = True, True
            
            # test if it is dependent of x1
            f_grid = f_constraints(np.array([x1s, x1[0]]))
            
            if equality:
                x2 = f_grid[0]
            else:
                x2 = f_grid[2]
            
            if n_constraints > 1:
                x2 = x2[constraint]
            
            x2 = sym.solve(x2, x1s)
            
            if len(x2) == 0:
                depends_on_x1 = False
            
            # test if it is dependent of x2
            f_grid = f_constraints(np.array([x1[0], x2s]))
            
            if equality:
                x2 = f_grid[0]
            else:
                x2 = f_grid[2]
            
            if n_constraints > 1:
                x2 = x2[constraint]
            
            x2 = sym.solve(x2, x2s)
            
            if len(x2) == 0:
                depends_on_x2 = False
            
            for i in range(x1.shape[0]):
                if not depends_on_x2:
                    f_grid = f_constraints(np.array([x2s, 0]))
                else:
                    f_grid = f_constraints(np.array([x1[i], x2s]))
                
                # get the value of x2
                if equality:
                    x2 = f_grid[0]
                else:
                    x2 = f_grid[2]
                
                if n_constraints > 1:
                    x2 = x2[constraint]
                
                # for all results of x2, associate x1 and y values
                x2 = sym.solve(x2, x2s)
                x2_temp = np.array([])
                x1_temp = np.array([])
                yy_temp = np.array([])
                for j in range(len(x2)):
                    if not x2[j].is_real:
                        break
                    x2_value = float(x2[j])
                    
                    if not depends_on_x2:
                        yy_temp = np.append(yy_temp, f_obj([x2_value, 0])[0])
                        x2_temp = np.append(x2_temp, x1[i])
                        x1_temp = np.append(x1_temp, x2_value)
                    else:
                        x1_temp = np.append(x1_temp, x1[i])
                        x2_temp = np.append(x2_temp, x2_value)
                        yy_temp = np.append(yy_temp, f_obj([x1[i], x2_value])[0])
                
                if not x2[j].is_real:
                    continue
                
                if not xx1.size:
                    xx1 = np.array([]).reshape(0, x1_temp.size)
                    
                if not xx2.size:
                    xx2 = np.array([]).reshape(0, x2_temp.size)
                    
                if not yy.size:
                    yy = np.array([]).reshape(0, yy_temp.size)
                
                x1_temp = np.array([x1_temp]).T
                x2_temp = np.array([x2_temp]).T
                yy_temp = np.array([yy_temp]).T
                
                xx1 = np.hstack([xx1, x1_temp]) if xx1.size else x1_temp
                xx2 = np.hstack([xx2, x2_temp]) if xx2.size else x2_temp
                yy = np.hstack([yy, yy_temp]) if yy.size else yy_temp
            
            # xx1 = np.ravel(xx1)
            # xx2 = np.ravel(xx2)
            # yy = np.ravel(yy)
            
            for row in range(xx1.shape[0]):
                h, = ax.plot(xx1[row, :], xx2[row, :], linestyle=linestyle_tuple[constraint][1], color=cor, label=label+str(constraint+1))
            handles.append(h)
        ax.legend(handles=handles)
            
    # Plot equality constraints
    if plot_h and f_constraints([X1[0,0], X2[0,0]])[0].size > 0:
        plot_constraints(equality=True)
    
    # Plot inequality constraints
    if plot_g and f_constraints([X1[0,0], X2[0,0]])[2].size > 0:
        plot_constraints(equality=False)
    
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((xmin, xmax))
    
    plt.show()
