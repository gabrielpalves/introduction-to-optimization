#!/usr/bin/env python
# coding: utf-8

# # Optimization method : Exterior Penalty Function (Section 5.7.1 Haftka)
# 
# ## For the unconstrained search: conjugated gradient + interval reduction method
# 
# 1) Step size: Golden Search Method, employing the function "minimize_scalar" from scipy.optimize
# 
# 2) Search direction : Conjugated Gradient, $\mathbf{d}_{(t)} = -\nabla_{\mathbf{x}} f_{(t)} + \beta_{(t)}\mathbf{d}_{(t-1)}$, onde $\beta_{(t)}=\left[\frac{||\nabla_{\mathbf{x}} f_{(t)}||}{||\nabla_{\mathbf{x}} f_{(t-1)}||}\right]^2$ 
# 
# The first step consists in defining the algorithms parameters, such as initial point $\mathbf{x}_{(0)}$, $\alpha_{(t)}$ and convergence tolerance constant $\epsilon_{\nabla}$, as well as the function to be minimized and its gradient evaluation:
# 
# help: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html

# In[48]:


import numpy as np
#from scipy.optimize import minimize_scalar
from plot_descent_penalty import plot_2d_contour, plot_3d_surface
from scipy.optimize import linprog



# Problem to be solved and variable for computational cost computation
global problem, r
problem = 6 # Defining the problem
cost_f, cost_g = 0, 0 # Defining the costs
# Initial guess
x=np.array([.5, .5]) #Starting point
# Bound constant for each linear programming step
cte = .025 #Step size

# Convergence Tolerance
TolG=1e-5
Tolf=1e-3
# Maximum number of iterations
itmax=50

#%% Define the objective function to be minimized and its constraints (it must be done by the user):

# Definition of the equation to be minimized
def f_obj(x):
    global problem
    
    if problem==1:
        f = x[0]**2+10*x[1]**2
        df = np.array([2*x[0], 20*x[1]])
    elif problem==2:
        f=(x[0]-1.5)**2+(x[1]-1.5)**2
        df=np.array([2*(x[0]-1.5), 2*(x[1]-1.5)])
    elif problem==3:
        f=(x[0]-1.5)**2+(x[1]-1.5)**2
        df=np.zeros((1,2))
        df[0,0]=2*(x[0]-1.5)
        df[0,1]=2*(x[1]-1.5)
    elif problem==4:
        f=(x[0]-1)**2+(x[1]-1)**2
        df=np.array([2*(x[0]-1), 2*(x[1]-1)])
    elif problem==5:
        f=x[0]+x[1]
        df=np.array([1, 1])
    elif problem == 6:
        f = x[0]**2 + x[1]**2 - 3*x[0]*x[1]
        df = np.array([2*x[0] - 3*x[1], 2*x[1] - 3*x[0]])
    return f, df

# Definition of the constraints: h and g
def nlconstraints(x):
    global problem
    
    if problem==1:
        h= x[0]+x[1]-4
        dh = np.array([1, 1])
        g=np.array([])
        dg=np.array([])
    elif problem==2:
        h=x[0]+x[1]-2
        dh=np.array([1, 1])
        g=np.array([])
        dg=np.array(np.zeros(x.shape))
    elif problem==3:
        h=np.array([])
        dh=np.array(np.zeros(x.shape))
        g=np.array([x[0]+x[1]-2])
        dg=np.array([[1, 1]])
    elif problem==4:
        h=np.array([])
        dh=np.array(np.zeros(x.shape))
        g=np.array([x[0]+x[1]-4, 2-x[0]])
        dg=np.array([[1, 1],[-1,0]])
    elif problem==5:
        h=np.array([])
        dh=np.array(np.zeros(x.shape))
        g=np.array([-(x[0]**2*x[1]/20-1), -(1/30*(x[0]+x[1]-5)**2 + 1/120*(x[0]-x[1]-12)**2 - 1), -x[0], -x[1]])
        dg=np.array([
            [-2*x[0]*x[1]/20, -x[0]**2/20],
            [ -(2/30*(x[0]+x[1]-5) + 2/120*(x[0]-x[1]-12)),  -(2/30*(x[0]+x[1]-5) - 2/120*(x[0]-x[1]-12))],
            [-1,0],
            [0,-1]
            ])
    elif problem == 6:
        h = np.array([])
        dh = np.array([])
        g = np.array([1/6*x[0]**2 + 1/6*x[1]**2 -1, -x[0], -x[1]])
        dg = np.array([
            [1/3*x[0], 1/3*x[1]],
            [-1, 0],
            [0, -1]
        ])
    return h, dh, g, dg

# From here on, the method is user independent:

# *** NÃO ESTÁ AUTOMATIZADO, ESTÁ TRAVADO PARA PROBLEMA DE 2 DIMENSÓES, precisa arrumar bounds...

def linearization(xk,cte):
    f,df=f_obj(xk)
    h, dh, g, dg = nlconstraints(xk)
    # Define c as the gradient of f
 
    # Assembles the matrices A and b
    b = -g
    A = dg#np.transpose(dg)
    #Define os bounds da iteração k como cte vezes o valor de xk
    lb=-cte*xk
    ub=cte*xk
    x0_bounds=(lb[0],ub[0])
    x1_bounds=(lb[1],ub[1])
    #Resolve o problema de programação linear
    res = linprog(c=df, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds])
    return res

#%% ITERATIVE SCHEME: SQL

k=0
res=linearization(x,cte)
d=res.x
stop = 0

points = [x]
values = [f_obj(x)[0]]

while stop==0:
    
    xk=x+d
    res=linearization(xk,cte)
    cost_f += 1
    cost_g += 1
    d=res.x
    f=res.fun
    b=res.slack
    points.append(xk)
    value, _ = f_obj(xk)
    values.append(value)

    # Update design and iteration number
    x=xk
    k=k+1
    # Stopping criteria
    if abs(f) < Tolf and abs(b.max())<TolG: 
        stop=1
    elif k>itmax:
        stop=1 
            
    
#%% The optimum design is stored in the variable $x$. Once the results are obtained, we may print them:

fopt,dfopt=f_obj(x)
hopt,dhopt,gopt,dgopt=nlconstraints(x)
print('Optimum found:')
print(x)
print('Objective function value at the optimum:')
print(fopt)
print('Inequality constraints at the optimum:')
print(gopt)
print('Equality constraints at the optimum:')
print(hopt)

print('Number of times that the f_obj function and constraints were evaluated, respectively:')
print(cost_f)
print(cost_g)
print('Number of iterations of the SLP method:')
print(k)


N = itmax # number of points to plot

#print(f_obj)

plot_2d_contour(points[:N], f_obj, plot_h=True, plot_g=True, f_constraints=nlconstraints)
plot_3d_surface(points[:N], values[:N], f_obj, plot_h=True, plot_g=True, f_constraints=nlconstraints)