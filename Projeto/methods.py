import numpy as np


def get_values(x, obj_fun, eq_cons, ineq_cons):
    """Get the values of the objective function,
    equality constraints, inequality constraints
    and all its gradients

    Args:
        x (numpy array): point (x1, x2)
        obj_fun (dict): dictionary containing lambda expressions that
            calculates the objective function and its gradient
        eq_cons (dict): contains lambda expressions that calculates
            the equality constraints and its gradient
        ineq_cons (dict): contains lambda expressions that calculates
            the equality constraints and its gradient

    Returns:
        values of gradients
    """

    f, df = obj_fun['fun'](x), obj_fun['jac'](x)
    h, dh = eq_cons['fun'](x), eq_cons['jac'](x)
    g, dg = ineq_cons['fun'](x), ineq_cons['jac'](x)

    return f, df, h, dh, g, dg


def methods(ext_pen_method, obj_fun, eq_cons, ineq_cons):
    match ext_pen_method:
        case 1:
            method = 'Quadratic Penalty'

            def phi(x, args):
                mu = args[0]
                f, df, h, dh, g, dg = get_values(
                    x, obj_fun, eq_cons, ineq_cons)

                auxg = np.maximum(0, g)
                ph = f + mu*(h.sum()**2 + auxg.sum()**2)

                # Construction of the gradient of phi: contribution of the
                # inequality constraints (g_i)
                dgaux = np.array(np.zeros(x.shape))
                # split into two situations:
                if g.size == 1:  # with only one constraints
                    dgaux = dg*np.maximum(0, g)
                else:
                    for i in range(g.size):  # and more constraints
                        dgaux = dgaux + dg[i, :]*np.maximum(0, g[i])

                # Construction of the gradient of phi: contribution of the
                # equality constraints (h_j)
                dhaux = np.array(np.zeros(x.shape))

                # split into two situations:
                if h.size == 1:  # with only one constraints
                    dhaux = dh*h
                else:
                    for j in range(h.size):  # and more constraints
                        dhaux = dhaux + dh[j, :]*h[j]

                # Gradient of phi:
                dph = df + 2*mu*(dhaux + dgaux)

                return ph, dph

        case 2:
            method = 'Augmented Lagrangian'

            def phi(x, args):
                mu = args[0]
                lambda_eq, lambda_ineq = args[1], args[2]
                f, df, h, dh, g, dg = get_values(
                    x, obj_fun, eq_cons, ineq_cons)

                # transform in matrix
                if dh.size > 0 and len(dh.shape) == 1:
                    dh = np.array([dh])

                if dg.size > 0 and len(dg.shape) == 1:
                    dg = np.array([dg])

                ghat = np.copy(g)
                for i in range(g.size):
                    c = -lambda_ineq[i]/mu
                    if g[i] < c:
                        ghat[i] = c
                        dg[i, :] = 0

                # maxg = np.max(0, ghat)
                L = f \
                    + np.dot(lambda_eq, h) + np.dot(lambda_ineq, ghat) \
                    + mu/2 * (
                        np.linalg.norm(h, ord=2)**2 +
                        np.linalg.norm(ghat, ord=2)**2
                        )

                dL = df \
                    + np.matmul(lambda_eq, dh) + np.matmul(lambda_ineq, dg) \
                    + mu*(np.matmul(h, dh) + np.matmul(ghat, dg))

                return L, dL

    return phi, method
