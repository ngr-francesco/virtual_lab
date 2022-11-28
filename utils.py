import numpy as np
import matplotlib.pyplot as plt
from math import floor

def get_time_unit_in_seconds(time_unit):
    min = ["min","mins","m","minutes"]
    sec = ["sec","s","secs","seconds"]
    hour = ["hour","h","hours"]
    if time_unit in min:
        return 60
    elif time_unit in sec:
        return 1
    elif time_unit in hour:
        return 3600
    else:
        raise ValueError(f"The selected time unit is not recognised {time_unit}. The available time unit names are:"
        f"{min} for minutes, {sec} for seconds or {hour} for hours.")

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

def bd_step_fast(n,gamma,mu,dt = 1):
    r1,r2 = np.random.rand(2).tolist()
    alpha = n * mu + gamma
    tau = 1/alpha*np.log(1/r1)
    n = n+1 if r2<gamma/alpha else n-1
    return n,floor(tau)

def bd_process_fast(*args, seed = None, **kwargs):
    """Optimized birth-death process to reduce computation time.
    WARNING: Don't use this version if you're intending to have complicated time dependencies on the parameters.
            In that case use the simpler bd_process function.
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
        # Set the seed
    if not seed is None:
        np.random.seed(seed)
    assert gamma.shape == mu.shape
    n = np.zeros(gamma.shape)
    # run the birth death 
    T = []
    if len(np.nonzero(gamma-gamma[0])[0]):
        k = np.nonzero(gamma-gamma[0])[0][0]
        T.append(k)
        g_r = np.append(np.zeros(k), gamma[k:])
        while k < len(gamma):
            tmp = np.append(np.zeros(k),gamma[k]*np.ones(len(gamma)-k))
            try:
                k = np.nonzero(g_r - tmp)[0][0]
                g_r = np.append(np.zeros(k),gamma[k:])
                T.append(k)
            except IndexError:
                break
    T.append(len(n)-1) # -1 is due to the expression in the except below
    n = np.zeros(gamma.shape)
    for idx,t_stop in enumerate(T):
        t = 0 if idx == 0 else T[T.index(t_stop)-1]
        while t<t_stop:
            n_new,tau = bd_step_fast(n[t],gamma[t],mu[t])
            try:
                n[t+1:t+tau]= n[t]
                n[t+tau] = n_new
            except IndexError:
                n[t+1:t_stop+1] = n[t]
            t += tau
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
    # Set the seed
    if not seed is None:
        np.random.seed(seed)
    assert gamma.shape == mu.shape
    # run the birth death process
    n = np.zeros(gamma.shape)
    for k in range(1,len(n)):
        n[k] = bd_step(n[k-1],gamma[k],mu[k],dt)
    return n 

def preprocess_experimental_data(data,datasheet_name = "experimental_data.json"):
    """
    This function is mainly to manage my specific data, so it's useless for data formatted in any other way.
    """
    import json
    new_datasheet = {}
    for d in data:
        if d == "info" or d == "renorm_var" or d == "time_unit":
            new_datasheet[d] = data[d]
            continue
        # Here I add 5 minutes because that's by default when stimuli happen for us
        # TODO: it's terrible, you should just start the simulation at -5 mins and then stimulate at 0
        x_data = (np.cumsum(data[d]["deltax"])+5).tolist()
        y_data = data[d]["y"].tolist()
        new_datasheet[d] = {"x": x_data,"y": y_data}
    with open(datasheet_name,"w+") as file:
        json.dump(new_datasheet,file)

def select_unique_handles_and_labels(handles,labels):
    """
    Just a helper function to make sure that the legend has all the necessary labels.
    Since handles and labels are connected I can't just use set(labels), so I use the labels as an indicator
    that tells me which handles to actually use.
    """
    h,l = [],[]
    for handle,label in zip(handles,labels):
        if not label in l:
            h.append(handle)
            l.append(label)
    return h,l
    
