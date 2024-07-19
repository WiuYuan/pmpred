import numpy as np
from scipy.stats import geninvgauss
from joblib import Parallel, delayed


def prscs_auto_subprocess_beta(subinput):
    (D, sigma2, N, psi, beta_hat) = subinput
    D = D.toarray()
    cov = sigma2 / N * np.linalg.inv(D + np.diag(1 / psi))
    mu = np.linalg.solve(D + np.diag(1 / psi), beta_hat)
    curr_beta = np.random.multivariate_normal(mu, cov)
    return curr_beta


def prscs_auto_subprocess_sigma2(subinput):
    curr_beta, beta_hat, D, psi = subinput
    sigma2_para = -2 * np.dot(curr_beta, beta_hat) + np.dot(
        curr_beta,
        D.toarray() @ curr_beta + curr_beta / psi,
    )
    sum_beta = np.dot(curr_beta, curr_beta / psi)
    return sigma2_para, sum_beta


def prscs_auto_subprocess_psi(subinput):
    N, sigma2, curr_beta, a, delta = subinput
    khi = N / sigma2 * curr_beta**2
    psi = geninvgauss.rvs(a - 1 / 2, np.sqrt(2 * delta * khi)) * np.sqrt(
        khi / (2 * delta)
    )
    psi[psi > 1] = 1
    return psi


def prscs_auto_subprocess_delta(subinput):
    a, b, psi, phi = subinput
    return np.random.gamma(a + b, 1 / (psi + phi))


def prscs_auto(PM, snplist, sumstats, para):
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
    beta_prscs = []
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
                    PM[i]["LD"][snplist[i]["index"]][:, snplist[i]["index"]],
                    sigma2,
                    N[i],
                    psi[i],
                    beta_hat[i],
                )
            )
        curr_beta = Parallel(n_jobs=para["n_jobs"])(
            delayed(prscs_auto_subprocess_beta)(d) for d in subinput
        )
        subinput = []
        for i in range(len(PM)):
            subinput.append(
                (
                    curr_beta[i],
                    beta_hat[i],
                    PM[i]["LD"][snplist[i]["index"]][:, snplist[i]["index"]],
                    psi[i],
                )
            )
        results = Parallel(n_jobs=para["n_jobs"])(
            delayed(prscs_auto_subprocess_sigma2)(d) for d in subinput
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
            subinput.append((N[i], sigma2, curr_beta[i], a, delta[i]))
        psi = Parallel(n_jobs=para["n_jobs"])(
            delayed(prscs_auto_subprocess_psi)(d) for d in subinput
        )
        subinput = []
        for i in range(len(PM)):
            subinput.append((a, b, psi[i], phi))
        delta = Parallel(n_jobs=para["n_jobs"])(
            delayed(prscs_auto_subprocess_delta)(d) for d in subinput
        )
        sum_delta = 0
        for i in range(len(PM)):
            sum_delta += np.sum(delta[i])
        w = np.random.gamma(1, 1 / (1 + phi))
        phi = np.random.gamma(M * b + 1 / 2, 1 / (sum_delta + w))
        if k >= 0:
            for i in range(len(PM)):
                avg_beta[i] += curr_beta[i]
        print("Run prscs_auto step:", k, "phi:", phi, "w:", w)
    for i in range(len(PM)):
        beta_prscs.append(avg_beta[i] / para["num_iter"])
    return beta_prscs, {"phi": phi, "w": w}
