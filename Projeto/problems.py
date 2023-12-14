import numpy as np


def problems(problem):
    """Chosen problem by the user to optimize

    Args:
        problem (int): chosen problem

    Returns:
        dictionaries containing lambda expressions that calculate
        the objective function and the constraints
        and all its gradients
    """
    match problem:
        # Problem 1 - Example 5.4, P. 170
        # Joaquim R. R. A. Martins, Andrew Ning -
        # Engineering Design Optimization (2021)
        case 1:
            # Objective function
            obj_fun = {
                'type': 'obj',
                'fun': lambda x: np.array([x[0] + 2*x[1]]),
                'jac': lambda x: np.array([1, 2])
            }

            # Equality constraints
            eq_cons = {
                'type': 'eq',
                'fun': lambda x: np.array([]),
                'jac': lambda x: np.array([])
                }

            # Inequality constraints
            ineq_cons = {
                'type': 'ineq',
                'fun': lambda x: np.array([1/4*x[0]**2 + x[1]**2 - 1]),
                'jac': lambda x: np.array([1/2*x[0], 2*x[1]])
                }

        # Problem 2 - Example 4.27, P. 122
        # Jasbir S. Arora - Introduction to optimum design (2004)
        case 2:
            obj_fun = {
                'type': 'obj',
                'fun': lambda x: np.array([(x[0]-1.5)**2 + (x[1]-1.5)**2]),
                'jac': lambda x: np.array([2*(x[0]-1.5), 2*(x[1]-1.5)])
            }

            eq_cons = {
                'type': 'eq',
                'fun': lambda x: np.array([x[0] + x[1] - 2]),
                'jac': lambda x: np.array([1, 1])
                }

            ineq_cons = {
                'type': 'ineq',
                'fun': lambda x: np.array([]),
                'jac': lambda x: np.array([])
                }

        # Problem 3 - Example 4.31, P. 134
        # Jasbir S. Arora - Introduction to optimum design (2004)
        case 3:
            obj_fun = {
                'type': 'obj',
                'fun': lambda x: np.array([x[0]**2 + x[1]**2 - 3*x[0]*x[1]]),
                'jac': lambda x: np.array([2*x[0] - 3*x[1], 2*x[1] - 3*x[0]])
            }

            eq_cons = {
                'type': 'eq',
                'fun': lambda x: np.array([]),
                'jac': lambda x: np.array([])
                }

            ineq_cons = {
                'type': 'ineq',
                'fun': lambda x: np.array([x[0]**2 + x[1]**2 - 6]),
                'jac': lambda x: np.array([2*x[0], 2*x[1]])
                }

        # Problem 4 - Example 5.6, P. 183
        # Jasbir S. Arora - Introduction to optimum design (2004)
        case 4:
            obj_fun = {
                'type': 'obj',
                'fun': lambda x: np.array([
                    x[0]**2 + x[1]**2 - 2*x[0] - 2*x[1] + 2
                    ]),
                'jac': lambda x: np.array([2*x[0] - 2, 2*x[1] - 2])
            }

            eq_cons = {
                'type': 'eq',
                'fun': lambda x: np.array([]),
                'jac': lambda x: np.array([])
                }

            ineq_cons = {
                'type': 'ineq',
                'fun': lambda x: np.array([-2*x[0] - x[1] + 4,
                                           -x[0] - 2*x[1] + 4]),
                'jac': lambda x: np.array([
                    [-2, -1],
                    [-1, -2]
                ])
                }

        # Problem 5 - Example 5.1, P. 176
        # Jasbir S. Arora - Introduction to optimum design (2004)
        case 5:
            obj_fun = {
                'type': 'obj',
                'fun': lambda x: np.array([(x[0] - 10)**2 + (x[1] - 8)**2]),
                'jac': lambda x: np.array([2*x[0] - 20, 2*x[1] - 16])
            }

            eq_cons = {
                'type': 'eq',
                'fun': lambda x: np.array([]),
                'jac': lambda x: np.array([])
                }

            ineq_cons = {
                'type': 'ineq',
                'fun': lambda x: np.array([x[0] + x[1] - 12, x[0] - 8]),
                'jac': lambda x: np.array([
                    [1, 1],
                    [1, 0]
                ])
                }

        # Problem 6 - Example 10.6, P. 359
        # Jasbir S. Arora - Introduction to optimum design (2004)
        case 6:
            obj_fun = {
                'type': 'obj',
                'fun': lambda x: np.array([
                    2*x[0]**3 + 15*x[1]**2 - 8*x[0]*x[1] - 4*x[0]
                    ]),
                'jac': lambda x: np.array([
                    6*x[0]**2 - 8*x[1] - 4, 30*x[1] - 8*x[0]
                    ])
            }

            eq_cons = {
                'type': 'eq',
                'fun': lambda x: np.array([x[0]**2 + x[0]*x[1] + 1]),
                'jac': lambda x: np.array([2*x[0] + x[1], x[0]])
                }

            ineq_cons = {
                'type': 'ineq',
                'fun': lambda x: np.array([x[0] - 1/4*x[1]**2 - 1]),
                'jac': lambda x: np.array([1, 1/2*x[1]])
                }

    return obj_fun, eq_cons, ineq_cons
