def check_convergence(k, f_old, f, f_opt, it_max, epsilon1, epsilon2):
    """Check convergence

    Args:
        k (int): current iteration
        f_old (float): old value of phi
        f (float): current value of phi
        f_opt (float): current value of the objective function
        it_max (int): convergence criteria:
            Maximum number of iterations
        epsilon1 (float): convergence criteria:
            Magnitude of the penalty terms
        epsilon2 (float): convergence criteria:
            Change in value of the penalized objective function

    Returns:
        boolean: determines the continuation of the
            exterior penalty optimization
    """
    stop1 = abs((f - f_opt)/f_opt)

    stop2 = 1.0
    if k > 0:
        stop2 = abs((f - f_old)/f)

    k = k + 1

    stop = True
    if k >= it_max:
        print('Stopped due to the number of iterations')
    elif stop1 <= epsilon1:
        print('Stopped due to the small magnitude of the penalty terms')
    elif stop2 <= epsilon2:
        print('Stopped due to a small change in value of the\
              penalized objective function')
    else:
        stop = False

    return stop
