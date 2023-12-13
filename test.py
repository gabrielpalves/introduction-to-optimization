import numpy as np

def get_problem(problem):
    obj_fun = {
            'type': 'obj',
            'fun' : lambda x : np.array([x[0] + 2*x[1]]),
            'jac' : lambda x : np.array([1, 2])
        }
        
    # Equality constraints
    eq_cons = {
        'type': 'eq',
        'fun' : lambda x: np.array([]),
        'jac' : lambda x: np.array([])
        }

    # Inequality constraints
    ineq_cons = {
        'type': 'ineq',
        'fun' : lambda x: np.array([1/4*x[0]**2 + x[1]**2 - 1]),
        'jac' : lambda x: np.array([1/2*x[0], 2*x[1]])
        }
    return obj_fun, eq_cons, ineq_cons

problem = 1
obj_fun, eq_cons, ineq_cons = get_problem(problem)
x0 = np.array([1, 2])

o = obj_fun['fun']
print(o(x0))
print(obj_fun['fun'](x0))


def get_values(x):
    obj_fun, eq_cons, ineq_cons = get_problem(problem)
    
    f, df = obj_fun['fun'](x), obj_fun['jac'](x)
    h, dh = eq_cons['fun'](x), eq_cons['jac'](x)
    g, dg = ineq_cons['fun'](x), ineq_cons['jac'](x)
    
    return f, df, h, dh, g, dg


def get_method(method):
    # Definition of the penalized Lagrangian (transformed functional phi)
    match method:
        # Quadratic penalty
        case 1:
            def phi(x):
                f, df, h, dh, g, dg = get_values(x)
                auxg = np.maximum(0, g)
                ph = f + mu*(h.sum()**2 + auxg.sum()**2)
                
                # Construction of the gradient of phi: contribution of the inequality constraints (g_i)
                dgaux = np.array(np.zeros(x.shape))
                if g.size == 1:  # split into two situations: with only one constraints, and more constraints
                    dgaux = dg*np.maximum(0, g)
                else:
                    for i in range(g.size):
                        dgaux = dgaux + dg[i, :]*np.maximum(0, g[i])
                    
                # Construction of the gradient of phi: contribution of the equality constraints (h_j)
                dhaux = np.array(np.zeros(x.shape))
                if h.size == 1:  # split into two situations: with only one constraints, and more constraints
                    dhaux = dh*h
                else:
                    for j in range(h.size):
                        dhaux = dhaux + dh[j, :]*h[j]
                
                # Gradient of phi:
                dph = df + 2*mu*(dhaux + dgaux)
                
                return ph, dph
        
        # Augmented Lagrangian
        case 2:
            def phi(x):
                f, df, h, dh, g, dg = get_values(x)
                
                ghat = np.copy(g)
                for i in range(g.size):
                    c = -lambda_ineq[i]/mu
                    if g[i] < c:
                        ghat[i] = c

                # maxg = np.max(0, ghat)
                L = f + np.dot(lambda_eq, h) + np.dot(lambda_ineq, ghat) + mu/2 * (np.dot(h, h) + np.dot(ghat, ghat))
                
                # transform in matrix
                if dh.size > 0 and len(dh.shape) == 1:
                    dh = np.array([dh])
                    
                if dg.size > 0 and len(dg.shape) == 1:
                    dg = np.array([dg])
                    
                dL = df + np.matmul(lambda_eq, dh) + np.matmul(lambda_ineq, dg) + mu*(np.matmul(h, dh) + np.matmul(g, dg))
                
                return L, dL
    
    return phi

a = get_method(1)
mu = 0.2
lambda_eq = 0.0
lambda_ineq = 0.5

x = np.array([-2, 5])
get_values(x)

f, df = obj_fun['fun'](x), obj_fun['jac'](x)
h, dh = eq_cons['fun'](x), eq_cons['jac'](x)
g, dg = ineq_cons['fun'](x), ineq_cons['jac'](x)

print(a(x))