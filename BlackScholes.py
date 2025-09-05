import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt


# Params
S0 = 20
K = 21
T = 90/365
r = 0.12
sigma = 0.20
option_type = 'C'


def BlackScholes_model(S0, K, T, r, sigma, option_type='C'):
    'BlackScholes pricing Model for Option'
    d1 = (np.log(S0/K) + (r + (0.5 * sigma**2))*T) / (sigma  * np.sqrt(T))
    d2 = d1 - (sigma * np.sqrt(T))

    if option_type == 'C':
            price = (S0 * norm.cdf(d1)) - (K * np.exp(-r*T) * norm.cdf(d2))

    elif option_type == 'P':
            price = (K * np.exp(-r*T) * norm.cdf(-d2)) - (S0 * norm.cdf(-d1))
        
    else:
        raise ValueError('Enter either C or P for option_type')

    return price
        