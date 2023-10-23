import matplotlib.pyplot as plt
import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


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
    
    print(f.shape, x.shape)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
    # ax = Axes3D(fig)

    x1 = np.arange(-20, 20, 0.25)
    x2 = np.copy(x1)
    X1, X2 = np.meshgrid(x1, x2)
        
    y = np.zeros((X1.shape))
    for i in range(X1.shape[0]):
        for j in range(X2.shape[0]):
            f_grid = f_obj([X1[i,j], X2[i,j]])
            y[i,j] = f_grid[0]

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
    
    # Plot equality constraints
    # OU EU USO SYMBOLIC PARA RESOLVER RESTRIÇÕES DE SEGUNDO GRAU OU MAIS (VER CUSTO)
    # OU SÓ PROGRAMO PARA RESTRIÇÕES DE PRIMEIRO GRAU
    # TAMBÉM DA PARA PLOTAR UM HIPERPLANO OU A INTERSEÇÃO NA FUNÇÃO OBJETIVO
    if plot_h and f_constraints([X1[0,0], X2[0,0]])[0].size > 0:
        for i in range(x1.shape[0]):
            f_grid = f_constraints([x1[i], 0])
            x2 = -f_grid[0] # get the value of x2
            f_obj([x1[i]])
    
    # Plot inequality constraints
    
    
    # if plot_h and f_constraints([X1[0,0], X2[0,0]])[0].size > 0:
    #     y_h = np.zeros((X1.shape))
    #     for i in range(X1.shape[0]):
    #         for j in range(X2.shape[0]):
    #             f_grid = f_constraints([X1[i,j], X2[i,j]])
    #             y_h[i,j] = f_grid[0]
        
    #     surface = ax.plot_surface(X1, X2, y_h,
    #                       cmap=cm.PiYG,
    #                       rstride=1, cstride=1, alpha=0.6)
        
    #     fig.colorbar(surface, shrink=0.5, aspect=5)
    
    # if plot_g and f_constraints([X1[0,0], X2[0,0]])[2].size > 0:
    #     y_g = np.zeros((X1.shape))
    #     for i in range(X1.shape[0]):
    #         for j in range(X2.shape[0]):
    #             f_grid = f_constraints([X1[i,j], X2[i,j]])
    #             y_g[i,j] = f_grid[2]
        
    #     surface = ax.plot_surface(X1, X2, y_h,
    #                       cmap=cm.BrBG,
    #                       rstride=1, cstride=1, alpha=0.6)
        
    #     fig.colorbar(surface, shrink=0.5, aspect=5)
    
    plt.show()


def plot_2d_contour(x, f_obj):
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

    x1 = np.arange(-20, 20, 0.25)
    x2 = np.copy(x1)
    x1, x2 = np.meshgrid(x1, x2)
    
    y = np.zeros((x1.shape))
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            f_grid = f_obj([x1[i,j], x2[i,j]])
            y[i,j] = f_grid[0]

    contours = ax.contour(x1, x2, y, 30)
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

    plt.show()
