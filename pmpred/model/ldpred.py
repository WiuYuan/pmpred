import numpy as np
from scipy.sparse.linalg import spsolve, gmres
from scipy.sparse import csr_matrix, eye
from joblib import Parallel, delayed


def get_marginal_beta(sumstats):
    beta_marginal = []
    for i in range(len(sumstats)):
        beta_marginal.append(list(map(float, sumstats[i]["beta"])))
    return beta_marginal


def pmpred_Q_times(P, x, n, Pid, sigma2, para):
    y = np.zeros(P.shape[0])
    y[Pid] = x * np.sqrt(n)
    return sigma2 * np.sqrt(n) * gmres(P, y, rtol=para["rtol"])[0][Pid]


def ldpred_inf_subprocess(subinput):
    PM, snplist, beta_hat, N, sigma2, i, para = subinput
    if len(N) == 0:
        return np.array([])
    P = PM["precision"].copy()
    P[snplist["index"], snplist["index"]] += sigma2 * N
    if i % 100 == 0:
        print("ldpred_inf_subprocess block:", i)
    return (
        sigma2
        * np.sqrt(N)
        * (
            np.sqrt(N) * beta_hat
            - pmpred_Q_times(
                P, np.sqrt(N) * beta_hat, N, snplist["index"], sigma2, para
            )
        )
    )


def ldpred_inf(PM, snplist, sumstats, para):
    beta_inf_set = []
    m = 0
    for i in range(len(sumstats)):
        m += len(sumstats[i]["N"])
    subinput = []
    for i in range(len(PM)):
        N = np.array(sumstats[i]["N"]).astype(float)
        beta_se = np.array(sumstats[i]["beta_se"]).astype(float)
        beta = np.array(sumstats[i]["beta"]).astype(float)
        scale = np.sqrt(N * beta_se**2)
        beta_hat = beta / scale
        subinput.append((PM[i], snplist[i], beta_hat, N, m / para["h2"], i, para))
    results = Parallel(n_jobs=para["n_jobs"])(
        delayed(ldpred_inf_subprocess)(d) for d in subinput
    )
    for i in range(len(PM)):
        # LD = PM[i]["LD"][snplist[i]["index"]][:, snplist[i]["index"]]
        # LD += eye(LD.shape[0], format="csr") * (m / (para["h2"] * para["N"]))
        # beta_inf = spsolve(LD, beta_hat)
        N = np.array(sumstats[i]["N"]).astype(float)
        beta_se = np.array(sumstats[i]["beta_se"]).astype(float)
        scale = np.sqrt(N * beta_se**2)
        beta_inf_set.append(results[i] * scale)
    return beta_inf_set, {"h2": para["h2"]}


def ldpred_grid_subprocess(subinput):
    PM, snplist, beta_hat, dotprods, curr_beta, N, h2_per_var, inv_odd_p, para = (
        subinput
    )
    if len(N) == 0:
        return np.array([]), np.array([]), np.array([])
    LD = PM["LD"]
    if isinstance(LD[0], np.float64):
        LD = csr_matrix([[LD[0]]])
    m = len(beta_hat)
    mean = np.zeros(m)
    for j in range(m):
        res_beta_hat_j = beta_hat[j] - (dotprods[j] - curr_beta[j])
        C1 = h2_per_var * N[j]
        C2 = 1 / (1 + 1 / C1)
        C3 = C2 * res_beta_hat_j
        C4 = C2 / N[j]
        post_p_j = 1 / (1 + inv_odd_p * np.sqrt(1 + C1) * np.exp(-C3 * C3 / C4 / 2))
        diff = -curr_beta[j]
        if post_p_j > np.random.rand():
            curr_beta[j] = np.random.normal(C3, np.sqrt(C4))
            diff += curr_beta[j]
            mean[j] = C3 * post_p_j
        else:
            curr_beta[j] = 0
        if diff != 0:
            dotprods += LD[:, j] * diff
    return mean, curr_beta, dotprods


def ldpred_grid(PM, snplist, sumstats, para):
    p = para["p"]
    h2 = para["h2"]
    curr_beta = []
    avg_beta = []
    dotprods = []
    scale_size = []
    M = 0
    for i in range(len(sumstats)):
        m = len(sumstats[i]["beta"])
        M += m
        curr_beta.append(np.zeros(m))
        avg_beta.append(np.zeros(m))
        dotprods.append(np.zeros(m))

    for k in range(-para["burn_in"], para["num_iter"]):
        print("ldpred_grid step:", k, "p:", p, "h2:", h2)
        h2_per_var = h2 / (M * p)
        inv_odd_p = (1 - p) / p
        subinput = []
        for i in range(len(PM)):
            N = np.array(sumstats[i]["N"]).astype(float)
            beta_hat = np.array(sumstats[i]["beta"]).astype(float)
            scale_size.append(
                np.sqrt(N) * np.array(sumstats[i]["beta_se"]).astype(float)
            )
            subinput.append(
                (
                    PM[i],
                    snplist[i],
                    beta_hat / scale_size[i],
                    dotprods[i],
                    curr_beta[i],
                    N,
                    h2_per_var,
                    inv_odd_p,
                    para,
                )
            )
        results = Parallel(n_jobs=para["n_jobs"])(
            delayed(ldpred_grid_subprocess)(d) for d in subinput
        )
        for i in range(len(PM)):
            mean, curr_beta[i], dotprods[i] = results[i]
            if k >= 0:
                avg_beta[i] += mean
    beta_grid = []
    for i in range(len(sumstats)):
        if len(sumstats[i]["beta"]) == 0:
            beta_grid.append(np.array([]))
        else:
            beta_grid.append(avg_beta[i] / para["num_iter"])
    return beta_grid, {"p": p, "h2": h2}


def ldpred_auto_subprocess(subinput):
    PM, snplist, beta_hat, dotprods, curr_beta, N, h2_per_var, inv_odd_p, para = (
        subinput
    )
    if len(N) == 0:
        return np.array([]), np.array([]), np.array([]), 0, 0
    LD = PM["LD"]
    if isinstance(LD[0], np.float64):
        LD = csr_matrix([[LD[0]]])
    m = len(beta_hat)
    Mc_add = 0
    mean = np.zeros(m)
    for j in range(m):
        res_beta_hat_j = beta_hat[j] - (dotprods[j] - curr_beta[j])
        C1 = h2_per_var * N[j]
        C2 = 1 / (1 + 1 / C1)
        C3 = C2 * res_beta_hat_j
        C4 = C2 / N[j]
        post_p_j = 1 / (1 + inv_odd_p * np.sqrt(1 + C1) * np.exp(-C3 * C3 / C4 / 2))
        diff = -curr_beta[j]
        if post_p_j > np.random.rand():
            curr_beta[j] = np.random.normal(C3, np.sqrt(C4))
            diff += curr_beta[j]
            Mc_add += 1
            mean[j] = C3 * post_p_j
        else:
            curr_beta[j] = 0
        if diff != 0:
            dotprods += LD[:, j] * diff
    return mean, curr_beta, dotprods, np.dot(curr_beta, LD.dot(curr_beta)), Mc_add


def ldpred_auto(PM, snplist, sumstats, para):
    p = para["p"]
    h2 = para["h2"]
    curr_beta = []
    avg_beta = []
    dotprods = []
    scale_size = []
    M = 0
    for i in range(len(sumstats)):
        m = len(sumstats[i]["beta"])
        M += m
        curr_beta.append(np.zeros(m))
        avg_beta.append(np.zeros(m))
        dotprods.append(np.zeros(m))

    for k in range(-para["burn_in"], para["num_iter"]):
        Mc = 0
        print("ldpred_auto step:", k, "p:", p, "h2:", h2)
        h2_per_var = h2 / (M * p)
        inv_odd_p = (1 - p) / p
        subinput = []
        for i in range(len(PM)):
            N = np.array(sumstats[i]["N"]).astype(float)
            beta_hat = np.array(sumstats[i]["beta"]).astype(float)
            scale_size.append(
                np.sqrt(N) * np.array(sumstats[i]["beta_se"]).astype(float)
            )
            subinput.append(
                (
                    PM[i],
                    snplist[i],
                    beta_hat / scale_size[i],
                    dotprods[i],
                    curr_beta[i],
                    N,
                    h2_per_var,
                    inv_odd_p,
                    para,
                )
            )
        h2 = 0
        results = Parallel(n_jobs=para["n_jobs"])(
            delayed(ldpred_auto_subprocess)(d) for d in subinput
        )
        for i in range(len(PM)):
            mean, curr_beta[i], dotprods[i], h2_add, Mc_add = results[i]
            Mc += Mc_add
            h2 += h2_add
            if k >= 0:
                avg_beta[i] += mean
        p = np.random.beta(1 + Mc, 1 + M - Mc)
    beta_auto_set = []
    for i in range(len(sumstats)):
        if len(sumstats[i]["beta"]) == 0:
            beta_auto_set.append(np.array([]))
        else:
            beta_auto_set.append(avg_beta[i] / para["num_iter"] * scale_size[i])
    return beta_auto_set, {"p": p, "h2": h2}
