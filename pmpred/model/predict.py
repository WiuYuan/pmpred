from scipy.stats import linregress
import numpy as np


def check_r_squared_phe(phestats, beta):
    y = phestats["Phenotype"].astype(float)
    x = phestats["X"] @ np.array(beta).astype(float)
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2


def check_r_squared_beta(beta1, beta2):
    slope, intercept, r_value, p_value, std_err = linregress(beta1, beta2)
    return r_value**2


def normalize(phestats):
    mean = phestats["X"].mean(axis=0)
    stddev = phestats["X"].std(axis=0)
    phestats["X"] = (phestats["X"] - mean) / stddev


def check_h2(PM, beta):
    h2 = 0
    for i in range(len(PM)):
        h2 += np.dot(beta[i], PM[i]["LD"].toarray() @ beta[i])
    return h2
