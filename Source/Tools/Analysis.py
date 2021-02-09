"""
Module with analysis functions to be used online in Fishualizer.py
"""
import numpy as np

def correlation_1D2D(Zecording_class, parent, **kwargs): # correlation function from Rémi (local.py)
    """
    Pearson coefficient of correlation between the calcium signals of two neurons
    Calculated manually to be faster in a 1 vector x 1 matrix

    Parameters:
    ----------
        Zecording_class: the class of Zecording
            everything can be accessed of the Zecording class
        parent: the parent of the Zecording_class (Fishualizer)
            everything can be accessed of the Fishualizer class
        **kwargs:
            sig_1D: str
                name of single signal
            sigs_2D: str
                name of multiple signals

    Returns:
    ----------
        r: float
            Coefficient of correlation
    """
    kwargs_keys = list(kwargs.keys())
    df1 = getattr(Zecording_class, kwargs[kwargs_keys[0]]) # sig_1D
    parent.statusBar().showMessage("Transposing")
    df2 = np.transpose(getattr(Zecording_class, kwargs[kwargs_keys[1]])) # sigs_2D.transpose()
    cov = np.dot(df1 - df1.mean(), df2 - df2.mean(axis=0)) / (df2.shape[0] - 1)
    # ddof=1 necessary because covariance estimate is unbiased (divided by n-1)
    p_var = np.sqrt(np.var(df1, ddof=1) * np.var(df2, axis=0, ddof=1))
    r = cov / p_var
    return r
