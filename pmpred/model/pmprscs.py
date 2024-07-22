import numpy as np
from scipy.stats import geninvgauss
from joblib import Parallel, delayed
from scipy.sparse.linalg import lgmres, gmres


def sparse_LD_times(P, x, Pidn0, para):
    if len(Pidn0) == 0:
        return np.array([])
    y = np.zeros(P.shape[0])
    y[Pidn0] = x
    return gmres(P, y, rtol=para["rtol"])[0][Pidn0]


def pmprscs_Q_times(P, x, psi, Pid, para):
    if len(Pid) == 0:
        return np.array([])
    y = np.zeros(P.shape[0])
    y[Pid] = x * np.sqrt(psi)
    return np.sqrt(psi) * gmres(P, y, rtol=para["rtol"])[0][Pid]


def pmprscs_auto_subprocess_beta(subinput):
    (PM, snplist, sigma2, N, psi, beta_hat, i, k, para) = subinput
    if len(N) == 0:
        return np.array([])
    P = PM["precision"].copy()
    P[snplist["index"], snplist["index"]] += psi
    mu = beta_hat * psi - np.sqrt(psi) * pmprscs_Q_times(
        P, np.sqrt(psi) * beta_hat, psi, snplist["index"], para
    )
    curr_beta = np.random.randn(len(N)) / np.sqrt(np.sqrt(N))
    x = curr_beta.copy()
    l = 1
    while np.max(np.abs(x)) / np.max(np.abs(curr_beta)) > para["taylor_rtol"]:
        x = -(0.5 - l + 1) / l * pmprscs_Q_times(P, x, psi, snplist["index"], para)
        curr_beta += x
        l += 1
    if i % 137 == 0:
        print("Run pmprscs_auto beta block:", i, "Iteration:", k, "Taylor:", l)
        print(max(curr_beta))
    return curr_beta * np.sqrt(sigma2) / np.sqrt(np.sqrt(N)) + mu


def pmprscs_auto_subprocess_sigma2(subinput):
    curr_beta, beta_hat, PM, snplist, psi, i, k, para = subinput
    sigma2_para = -2 * np.dot(curr_beta, beta_hat) + np.dot(
        curr_beta,
        sparse_LD_times(PM["precision"], curr_beta, snplist["index"], para)
        + curr_beta / psi,
    )
    sum_beta = np.dot(curr_beta, curr_beta / psi)
    if i % 137 == 0:
        print("Run pmprscs_auto sigma2 block:", i, "Iteration:", k)
    return sigma2_para, sum_beta


def pmprscs_auto_subprocess_psi(subinput):
    N, sigma2, curr_beta, a, delta = subinput
    khi = N / sigma2 * curr_beta**2
    psi = geninvgauss.rvs(a - 1 / 2, np.sqrt(2 * delta * khi)) * np.sqrt(
        khi / (2 * delta)
    )
    psi[psi > 1] = 1
    return psi


def pmprscs_auto_subprocess_delta(subinput):
    a, b, psi, phi = subinput
    return np.random.gamma(a + b, 1 / (psi + phi))


def pmprscs_auto(PM, snplist, sumstats, para):
    M = 0
    a = para["prscs_a"]
    b = para["prscs_b"]
    curr_beta = []
    avg_beta = []
    beta_hat = []
    N = []
    N_total = 0
    scale_size = []
    delta = []
    psi = []
    phi = 1
    sigma2 = 1
    beta_pmprscs = []
    for i in range(len(PM)):
        M += len(sumstats[i]["beta"])
    for i in range(len(PM)):
        beta_hat.append(np.array(sumstats[i]["beta"]).astype(float))
        N.append(np.array(sumstats[i]["N"]).astype(float))
        N_total += np.sum(N[i])
        scale_size.append(
            np.sqrt(N[i] * np.array(sumstats[i]["beta_se"]).astype(float) ** 2)
        )
        beta_hat[i] = beta_hat[i] / scale_size[i]
        m = len(beta_hat[i])
        curr_beta.append(np.zeros(m))
        delta.append(np.random.gamma(b, 1 / phi, size=m))
        psi.append(np.random.gamma(a, 1 / delta[i]))
        avg_beta.append(np.zeros(m))
        snplist[i]["index"] = np.array(snplist[i]["index"])
    N_total = N_total / M
    for k in range(-para["burn_in"], para["num_iter"]):
        subinput = []
        for i in range(len(PM)):
            subinput.append(
                (
                    PM[i],
                    snplist[i],
                    sigma2,
                    N[i],
                    psi[i],
                    beta_hat[i],
                    i,
                    k,
                    para,
                )
            )
        curr_beta = Parallel(n_jobs=para["n_jobs"])(
            delayed(pmprscs_auto_subprocess_beta)(d) for d in subinput
        )
        subinput = []
        for i in range(len(PM)):
            subinput.append(
                (
                    curr_beta[i],
                    beta_hat[i],
                    PM[i],
                    snplist[i],
                    psi[i],
                    i,
                    k,
                    para,
                )
            )
        results = Parallel(n_jobs=para["n_jobs"])(
            delayed(pmprscs_auto_subprocess_sigma2)(d) for d in subinput
        )
        sigma2_para = 0
        sum_beta = 0
        for i in range(len(PM)):
            sigma2_para += results[i][0]
            sum_beta += results[i][1]
        err = max(N_total / 2 * (1 + sigma2_para), N_total / 2 * sum_beta)
        sigma2 = 1 / np.random.gamma((N_total + M) / 2, 1 / err)
        subinput = []
        for i in range(len(PM)):
            subinput.append((a, b, psi[i], phi))
        delta = Parallel(n_jobs=para["n_jobs"])(
            delayed(pmprscs_auto_subprocess_delta)(d) for d in subinput
        )
        subinput = []
        for i in range(len(PM)):
            subinput.append((N[i], sigma2, curr_beta[i], a, delta[i]))
        psi = Parallel(n_jobs=para["n_jobs"])(
            delayed(pmprscs_auto_subprocess_psi)(d) for d in subinput
        )
        sum_delta = 0
        for i in range(len(PM)):
            sum_delta += np.sum(delta[i])
        w = np.random.gamma(1, 1 / (1 + phi))
        phi = np.random.gamma(M * b + 1 / 2, 1 / (sum_delta + w))
        if k >= 0:
            for i in range(len(PM)):
                avg_beta[i] += curr_beta[i]
        print("Run pmprscs_auto step:", k, "phi:", phi, "w:", w)
    for i in range(len(PM)):
        beta_pmprscs.append(avg_beta[i] / para["num_iter"])
    return beta_pmprscs, {"phi": phi, "w": w}
