import numpy as np
import scipy.optimize


def entropy(p, *args):
    return sum(p*np.log(p))

def sum_constraint(p, *args):
    return sum(p) - 1

def mean_constraint(p, *args):
    mean = args[0]
    x = np.arange(1,7)
    return p.dot(x) - mean

def analytical(l):
    x = np.arange(1,7)
    p = np.exp(-l*x)
    p /= sum(p)
    return p

def chi2_constraint(p, *args):
    cs_calc = args[0]
    cs_exp  = args[1]
    var     = args[2]
    cs_calc_avg = (cs_calc * p[:,None]).sum(0)
    return np.sum((cs_calc_avg - cs_exp)**2 / var) - cs_calc.shape[0]

def cs_example(cs_calc, cs_exp, var):
    p = np.ones(cs_calc.shape[0])
    p /= p.sum()

    constraints = [{'type':'eq',
                    'fun':sum_constraint},
                    {'type':'eq',
                    'fun':chi2_constraint,
                    'args':(cs_calc, cs_exp, var)}
                    ]
    x = scipy.optimize.minimize(entropy, p, bounds=[[1e-9,None] for _ in range(cs_calc.shape[0])],
            constraints=constraints, options={'maxiter':5000}, tol=1e-9)
    if not x.success:
        print("Bad convergence")
    return x.x

def cs_example2(cs_calc, cs_exp, var, hydrogens):
    p = np.ones(cs_calc.shape[0])
    p /= p.sum()

    constraints = [{'type':'eq',
                    'fun':sum_constraint},
                    {'type':'eq',
                    'fun':chi2_constraint,
                    'args':(cs_calc[:,hydrogens], cs_exp[hydrogens], var[hydrogens])},
                    {'type':'eq',
                    'fun':chi2_constraint,
                    'args':(cs_calc[:,~hydrogens], cs_exp[~hydrogens], var[~hydrogens])}
                    ]
    x = scipy.optimize.minimize(entropy, p, bounds=[[1e-9,None] for _ in range(cs_calc.shape[0])],
            constraints=constraints, options={'maxiter':5000}, tol=1e-9)
    if not x.success:
        print("Bad convergence")
    return x.x

def analytical(l):
    x = np.arange(1,7)
    p = np.exp(-l*x)
    p /= sum(p)
    return p

def dice_example_l():
    """
    Example with a biased dice with a previously observed mean
    The lagrange multiplier l is optimized rather than p
    """
    def entropy(l, *args):
        """
        Entropy of p
        """
        p = analytical(l)
        return sum(p*np.log(p))

    def mean_constraint(l, *args):
        """
        Constraint that the weighted mean is fixed
        """
        mean = args[0]
        x = np.arange(1,7)
        p = analytical(l)
        return p.dot(x) - mean

    def analytical(l):
        """
        Convert the lagrange multiplier l to p
        """
        x = np.arange(1,7)
        p = np.exp(-l*x)
        return p/sum(p)

    # Observed mean
    mean = 3
    # Starting guess
    l = -1
    # Set constraints
    constraints = [{'type':'eq',
                    'fun':mean_constraint,
                    'args': (mean,)}
                    ]

    opt = scipy.optimize.minimize(entropy, l, constraints=constraints)
    print(analytical(opt.x))

def dice_example_p():
    """
    Example with a biased dice with a previously observed mean
    p is optimized directly, rather than the lagrange multiplier l
    """
    def entropy(p, *args):
        """
        Entropy of p
        """
        return sum(p*np.log(p))

    def sum_constraint(p, *args):
        """
        Constraint that p sum to 1
        """
        return sum(p) - 1

    def mean_constraint(p, *args):
        """
        Constraint that the weighted mean is fixed
        """
        mean = args[0]
        x = np.arange(1,7)
        return p.dot(x) - mean

    # Observed mean
    mean = 3
    # Starting guess
    p = np.ones(6)/6
    # Set constraints
    constraints = [{'type':'eq',
                    'fun':sum_constraint},
                    {'type':'eq',
                    'fun':mean_constraint,
                    'args': (mean,)}
                    ]

    opt = scipy.optimize.minimize(entropy, p, bounds=[[1e-9,None] for _ in range(6)],
            constraints=constraints)
    print(opt.x)


if __name__ == "__main__":
    print(dice_example_p())
    print(dice_example_l())
