


def rmse(sim, obs):
    """
    Root Mean Squared Error
    """
    
    import numpy as np
    return np.sqrt(np.mean((sim-obs)**2))

def mae(sim,obs):
    """
    Mean Absolute Error
    """
    
    import numpy as np
    return np.mean(abs(sim-obs))

def mb(sim, obs):
    """
    Mean Bias

    """
    import numpy as np
    return np.mean(sim-obs)

def nse(sim, obs):
    """
    Nash Sutcliffe efficiency coefficient
  
    """
    import numpy
    return 1-sum((sim-obs)**2)/sum((obs-numpy.mean(obs))**2)

def corr(sim,obs):
    """
    correlation coefficient
    """
    import numpy as np
    if sim.size == 0:
        corr = np.NaN
    else:
        corr = np.corrcoef(obs, sim)[0,1]
        
    return corr