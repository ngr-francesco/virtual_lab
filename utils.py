import numpy as np
import matplotlib.pyplot as plt
__all__ = ['bd_process']

def bd_step(n,gamma,mu, dt = 1):
    if n == 0:
        p = gamma*dt
        q = 0
    else:
        p = (gamma)*dt
        q = (mu*n)*dt
    s = float(np.random.rand(1))
    choice = [s < p , s > p and s < p+q, s > p+q]
    val = np.array([1,-1,0])
    n += int(val[choice])
    return n

def bd_process(*args, dt = 1, seed = None, **kwargs):
    """
    Inputs:
        n : numpy.array of the desired length, initialization doesn't matter
        except for the first value.
        gamma: numpy.array with the time evolution of the birth prob
        mu: same but for death prob
        dt: ...
    """
    if len(args) > 2:
        raise ValueError("The bd_process takes maximum 2 positional arguments for gamma and mu.")
    if isinstance(args[0],list) and len(args[0]) == 2:
        gamma = args[0][0]
        mu = args[0][1]
    elif isinstance(args[0],np.ndarray) and len(args[0])> 2:
        gamma = args[0]
        mu = args[1]
    if len(args) == 0:
        msg = """The gamma and mu values must be given either through positional arguments (e.g. bd_process(gamma,mu)) 
         or as keyword arguments (e.g. bd_process(gamma = g, mu = m))"""
        gamma = kwargs.get('gamma',ValueError)
        mu = kwargs.get('mu',ValueError)
        if gamma is ValueError or mu is ValueError:
            raise ValueError(msg)
    if not seed is None:
        np.random.seed(seed)
    assert gamma.shape == mu.shape
    n = np.zeros(gamma.shape)
    for k in range(1,len(n)):
        n[k] = bd_step(n[k-1],gamma[k],mu[k],dt)
    return n 