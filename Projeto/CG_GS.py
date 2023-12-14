import numpy as np
from scipy.optimize import minimize_scalar


def f_alpha(alpha, args):
    """Definition of the equation to be minimized as function of
    the step size alpha

    Args:
        alpha (float): step size
        args (array): array with arguments point (x), direction of search (d)
            and function (phi)

    Returns:
        f (float): value of phi
    """
    x, d, phi = args[:3]

    xt = x + alpha*d
    f, _ = phi(xt, args[3:])

    return f


def CG_GS(x, alpha0, TolG, phi, f_alpha, mu, lambda_eq, lambda_ineq):
    """Conjugate gradient for search direction and
    Golden Search method to calculate step size

    Args:
        x (numpy array): starting point
        alpha0 (float): upper bound for Golden Search method
        TolG (float): convergence tolerance for Conjugate Gradient
        phi (method): transformed functional to be minimized
        f_alpha (method): phi with x plus a alpha step
        mu (float): penalty parameter
        lambda_eq (_type_): Lagrangian multipliers for equality constraints
        lambda_ineq (_type_): Lagrangian multipliers for inequality constraints

    Returns:
        x: optimal point
        f: optimal functional value
        df: gradient of phi at optimal
        t: number of iterations in the conjugate gradient
        xs: evaluated points during iteration (for plotting later)
        fs: all values of evaluated points (for plotting later)
    """

    # f and df values at the initial point
    [f, df] = phi(x, args=([mu, lambda_eq, lambda_ineq]))
    dftm1 = df

    # Initialize variables
    t, dtm1 = 1, 0
    xs = [x]
    fs = [f]

    while np.sqrt(df @ df) > TolG:
        # Search direction: Conjugated Gradient
        beta = (np.linalg.norm(df)/np.linalg.norm(dftm1))**2
        d = -df + beta*dtm1

        # Step determination: Golden Search (method='golden'),
        # Brent (method='brent') or Bounded (method='bounded')
        alpha = minimize_scalar(f_alpha, bounds=(.001, alpha0),
                                args=([x, d, phi, mu, lambda_eq, lambda_ineq]),
                                method='bounded')

        # Update the current point
        xt = x + alpha.x*d
        xs.append(xt)

        # Saves information of gradient and
        # descent direction of current iteration
        dftm1 = df
        dtm1 = d

        # Evaluate the objective function and gradient at the new point
        [f, df] = phi(xt, args=([mu, lambda_eq, lambda_ineq]))
        fs.append(f)

        # Update the design variable and iteration number
        x = xt
        t = t + 1 + alpha.nfev

    return x, f, df, t, xs, fs
