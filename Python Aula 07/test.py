import numpy as np
x = np.array([2, 3])
b = x[0] + x[1]
a = np.array([1, 1])
c = np.array([])

print(b.shape, a.shape, b.size, a.size, c.shape, c.size)

# x1 = np.arange(-20,20,1)
# print(x1.shape)

# x = np.array([1, -1])
# h = x[0]+x[1]-4
# dh = np.array([1, 1])
# g = np.array([])
# f = np.array([[1,2,3,4],[3,4,5,6]])

# print(x.shape, h.shape, dh.shape, f.shape)
# print(x.size, h.size, dh.size, f.size)

# if h.size == 0:
#     print('not even h')

# if g.size == 0:
#     print('not even g')

# if not h.shape:
#     print('not')
# else:
#     print('ok')

# if not x.shape:
#     print('not x')
    
# if len(h.shape) == 0:
#     print('length is zero')

# if isinstance(x, list):
#     x = np.array(x)

def nlconstraints(x):
    h = x[0]**2+x[1]-4
    dh = np.array([1, 1])
    
    return h, dh

# divide o resultado de 'h' pelo negativo da derivada

# testing symbolic math
import sympy as sym

abc = sym.Symbol('abc')
bcd = sym.Symbol('bcd')

a, _ = nlconstraints([abc, 1])

print(a)

result = sym.solve(a, abc)
print(result, type(result))
print(result[0], len(result))
print(float(result[0]))

#
#
#
problem = 2
def f_obj(x):
    if problem==1:
        f = x[0]**2+10*x[1]**2
        df = np.array([2*x[0], 20*x[1]])
    elif problem==2:
        f=(x[0]-1.5)**2+(x[1]-1.5)**2
        df=np.array([2*(x[0]-1.5), 2*(x[1]-1.5)])
    elif problem==3:
        f=(x[0]-1.5)**2+(x[1]-1.5)**2
        df=np.array([2*(x[0]-1.5), 2*(x[1]-1.5)])
    elif problem==4:
        f=(x[0]-1)**2+(x[1]-1)**2
        df=np.array([2*(x[0]-1), 2*(x[1]-1)])
    elif problem==5:
        L=5 # meters
        f=x[0]*x[1]*L
        df=np.array([(x[1]*L), (x[0]*L)])   
        
    return f, df

x = np.array([[3,3],[2,2]])
xmin = np.floor(-np.max(np.abs(x[0, :])) - 1)
xmax = np.ceil(np.max(np.abs(x[0, :])) + 1)
x1 = np.arange( xmin, xmax, 0.25 )
x2 = np.copy(x1)
X1, X2 = np.meshgrid(x1, x2)
    
y = np.zeros((X1.shape))
for i in range(X1.shape[0]):
    for j in range(X2.shape[0]):
        f_grid = f_obj([X1[i,j], X2[i,j]])
        y[i,j] = f_grid[0]


import matplotlib.pyplot as plt
from matplotlib import cm

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})

surface = ax.plot_surface(X1, X2, y,
                          cmap=cm.coolwarm,
                          rstride=1, cstride=1, alpha=0.6)

xx = np.array([-4, -4, -3.5, -3.5, -3, -3, -2.5, -2.5, -2, -2, -1.5, -1.5, -1, -1, -0.5, -0.5, 0, 0, 0.5, 0.5, 1, 1, 1.5, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4])
xy = np.array([])
yy = np.array([])
for i in range(0, xx.shape[0], 2):
    r = np.sqrt(-xx[i]+4)
    # here, xy (x2) is being found
    # so we need to plot following x2 (xy) order, not x1 (xx)
    # hence, the sort after the loop
    f_grid_1 = f_obj([xx[i], -r])
    f_grid_2 = f_obj([xx[i],  r])
    
    xy = np.append(xy, np.array([-r, r]))
    yy = np.append(yy, np.array([f_grid_1[0], f_grid_2[0]]))

idx = np.argsort(xy)
xy = xy[idx]
xx = xx[idx]
yy = yy[idx]

ax.plot(xx, xy, yy, markerfacecolor='k', markeredgecolor='k', marker='.', markersize=7)

XX, XY = np.meshgrid(xx, xy)
