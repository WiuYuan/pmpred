import numpy as np

# from scipy.sparse.linalg import cg
from joblib import Parallel, delayed


def cg(P, b, rtol, maxiter=1000):
    bnrm2 = np.linalg.norm(b)
    x = np.zeros(P.shape[0])
    if bnrm2 == 0:
        return x
    atol = bnrm2 * rtol
    r = b.copy()
    for iteration in range(maxiter):
        if np.linalg.norm(r) < atol:
            return x

        z = r.copy()
        rho_cur = np.dot(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:
            p = np.empty_like(r)
            p[:] = z[:]

        q = P.dot(p)
        alpha = rho_cur / np.dot(p, q)
        x += alpha * p
        r -= alpha * q
        rho_prev = rho_cur
    print("A")
    return x


def sparse_LD_times(P, x, Pidn0, para):
    if len(Pidn0) == 0:
        return np.array([])
    y = np.zeros(P.shape[0])
    y[Pidn0] = x
    return cg(P, y, rtol=para["rtol"])[Pidn0]


def pmpred_Q_times(P, x, n, Pid, sigma2, para):
    y = np.zeros(P.shape[0])
    y[Pid] = x * np.sqrt(n)
    return sigma2 * np.sqrt(n) * cg(P, y, rtol=para["rtol"])[Pid]


def pmpred_grid_subprocess(subinput):
    (
        PM,
        snplist,
        beta_hat,
        curr_beta,
        R_curr_beta,
        N,
        h2_per_var,
        inv_odd_p,
        h2,
        para,
        i,
        k,
    ) = subinput
    res_beta_hat = beta_hat * np.sqrt(1 - h2) - (R_curr_beta - curr_beta)
    n = N / (1 - h2)
    C1 = h2_per_var * n
    C2 = 1 / (1 + 1 / C1)
    C3 = C2 * res_beta_hat
    C4 = C2 / n
    m = len(beta_hat)
    post_p = 1 / (1 + inv_odd_p * np.sqrt(1 + C1) * np.exp(-C3 * C3 / C4 / 2))
    z = (np.random.rand(m) < post_p).astype(int)
    id0 = np.where(z == 0)[0]
    idn0 = np.where(z != 0)[0]
    Pidn0 = snplist["index"][idn0]
    mean_beta = np.zeros(m)
    if len(Pidn0) != 0:
        P = PM["precision"].copy()
        P[Pidn0, Pidn0] += h2_per_var * n[idn0]
        delta = np.sqrt(1 - h2) * beta_hat[idn0] * n[idn0]
        mean = h2_per_var * (
            delta - pmpred_Q_times(P, delta, n[idn0], Pidn0, h2_per_var, para)
        )
        curr_beta[idn0] = np.random.randn(len(idn0))
        x = curr_beta[idn0].copy()
        l = 1
        while np.max(np.abs(x)) / np.max(np.abs(curr_beta)) > para["taylor_rtol"]:
            x = (
                -(0.5 - l + 1)
                / l
                * pmpred_Q_times(P, x, n[idn0], Pidn0, h2_per_var, para)
            )
            curr_beta[idn0] += x
            l += 1
        if i % 137 == 0:
            print("Run ldpred_auto block:", i, "Iteration:", k, "Taylor:", l)
        curr_beta[idn0] *= np.sqrt(h2_per_var)
        curr_beta[idn0] += mean
        mean_beta[idn0] = mean
    curr_beta[id0] = 0
    R_curr_beta = sparse_LD_times(PM["precision"], curr_beta, snplist["index"], para)
    return curr_beta, R_curr_beta, mean_beta


def pmldpred_grid(PM, snplist, sumstats, para):
    p = para["p"]
    h2 = para["h2"]
    M = 0
    beta_pmldpred_auto = []
    curr_beta = []
    avg_beta = []
    beta_hat = []
    N = []
    scale_size = []
    R_curr_beta = []
    for i in range(len(PM)):
        M += len(sumstats[i]["beta"])
    for i in range(len(PM)):
        beta_hat.append(np.array(sumstats[i]["beta"]).astype(float))
        N.append(np.array(sumstats[i]["N"]).astype(float))
        scale_size.append(
            np.sqrt(N[i]) * np.array(sumstats[i]["beta_se"]).astype(float)
        )
        beta_hat[i] = beta_hat[i] / scale_size[i]
        m = len(beta_hat[i])
        curr_beta.append(np.zeros(m))
        R_curr_beta.append(np.zeros(m))
        avg_beta.append(np.zeros(m))
        snplist[i]["index"] = np.array(snplist[i]["index"])
    for k in range(-para["burn_in"], para["num_iter"]):
        print("Run pmldpred_auto step:", k, "p:", p, "h2:", h2)
        h2_per_var = h2 / (M * p)
        inv_odd_p = (1 - p) / p
        subinput = []
        for i in range(len(PM)):
            subinput.append(
                (
                    PM[i],
                    snplist[i],
                    beta_hat[i],
                    curr_beta[i],
                    R_curr_beta[i],
                    N[i],
                    h2_per_var,
                    inv_odd_p,
                    h2,
                    para,
                    i,
                    k,
                )
            )
        results = Parallel(n_jobs=para["n_jobs"])(
            delayed(pmpred_grid_subprocess)(d) for d in subinput
        )
        for i in range(len(PM)):
            curr_beta[i], R_curr_beta[i], mean_beta = results[i]
            if k >= 0:
                avg_beta[i] += mean_beta
    for i in range(len(PM)):
        beta_pmldpred_auto.append(avg_beta[i] / para["num_iter"])
        beta_pmldpred_auto[i] = beta_pmldpred_auto[i] * scale_size[i]
        snplist[i]["index"] = snplist[i]["index"].tolist()
    outpara = {
        "p": p,
        "h2": h2,
        "burn_in": para["burn_in"],
        "num_iter": para["num_iter"],
        "log10rtol": np.log(para["rtol"]) / np.log(10),
        "log10taylor_rtol": np.log(para["taylor_rtol"]) / np.log(10),
        "prop": para["prop"],
    }
    return beta_pmldpred_auto, outpara


def pmpred_auto_subprocess(subinput):
    (
        PM,
        snplist,
        beta_hat,
        curr_beta,
        R_curr_beta,
        N,
        h2_per_var,
        inv_odd_p,
        h2,
        para,
        i,
        k,
    ) = subinput
    # res_beta_hat = beta_hat * np.sqrt(1 - h2) - (R_curr_beta - curr_beta)
    # res_beta_hat = np.sqrt(1 - h2) * (beta_hat - R_curr_beta) + curr_beta
    res_beta_hat = beta_hat - (R_curr_beta - curr_beta)
    # n = N / (1 - h2)
    n = N
    C1 = h2_per_var * n
    C2 = 1 / (1 + 1 / C1)
    C3 = C2 * res_beta_hat
    C4 = C2 / n
    m = len(beta_hat)
    post_p = 1 / (1 + inv_odd_p * np.sqrt(1 + C1) * np.exp(-C3 * C3 / C4 / 2))
    z = (np.random.rand(m) < post_p).astype(int)
    id0 = np.where(z == 0)[0]
    idn0 = np.where(z != 0)[0]
    Pidn0 = snplist["index"][idn0]
    mean_beta = np.zeros(m)
    if len(Pidn0) != 0:
        P = PM["precision"].copy()
        P[Pidn0, Pidn0] += h2_per_var * n[idn0]
        delta = np.sqrt(1 - h2) * beta_hat[idn0] * n[idn0]
        # delta = (
        #     np.sqrt(1 - h2) * beta_hat[idn0] + (1 - np.sqrt(1 - h2)) * R_curr_beta[idn0]
        # ) * n[idn0]
        # delta = beta_hat[idn0] * n[idn0]
        mean = h2_per_var * (
            delta - pmpred_Q_times(P, delta, n[idn0], Pidn0, h2_per_var, para)
        )
        # mean = np.linalg.solve(
        #     PM["LD"][Pidn0][:, Pidn0].toarray() + np.diag(1 / (h2_per_var * n[idn0])),
        #     beta_hat[idn0],
        # )
        # alpha = np.zeros(P.shape[0])
        # alpha[Pidn0] = np.sqrt(1 - h2) * beta_hat[idn0] * np.sqrt(n[idn0])
        # mean = (
        #     h2_per_var
        #     * np.sqrt(n[idn0])
        #     * cg(P, PM["precision"].dot(alpha), rtol=para["rtol"])[0][Pidn0]
        # )
        curr_beta[idn0] = np.random.randn(len(idn0))
        x = curr_beta[idn0].copy()
        l = 1
        while np.max(np.abs(x)) / np.max(np.abs(curr_beta)) > para["taylor_rtol"]:
            x = (
                -(0.5 - l + 1)
                / l
                * pmpred_Q_times(P, x, n[idn0], Pidn0, h2_per_var, para)
            )
            curr_beta[idn0] += x
            l += 1
        if i % 137 == 0:
            print("Run ldpred_auto block:", i, "Iteration:", k, "Taylor:", l)
        curr_beta[idn0] *= np.sqrt(h2_per_var)
        curr_beta[idn0] += mean
        mean_beta[idn0] = mean
    curr_beta[id0] = 0
    R_curr_beta = sparse_LD_times(PM["precision"], curr_beta, snplist["index"], para)
    return curr_beta, R_curr_beta, len(idn0), np.dot(curr_beta, R_curr_beta), mean_beta


def pmldpred_auto(PM, snplist, sumstats, para):
    p = para["p"]
    h2 = para["h2"]
    M = 0
    beta_pmldpred_auto = []
    curr_beta = []
    avg_beta = []
    beta_hat = []
    N = []
    scale_size = []
    R_curr_beta = []
    for i in range(len(PM)):
        M += len(sumstats[i]["beta"])
    for i in range(len(PM)):
        beta_hat.append(np.array(sumstats[i]["beta"]).astype(float))
        N.append(np.array(sumstats[i]["N"]).astype(float))
        scale_size.append(
            np.sqrt(N[i]) * np.array(sumstats[i]["beta_se"]).astype(float)
        )
        beta_hat[i] = beta_hat[i] / scale_size[i]
        m = len(beta_hat[i])
        curr_beta.append(np.zeros(m))
        R_curr_beta.append(np.zeros(m))
        avg_beta.append(np.zeros(m))
        snplist[i]["index"] = np.array(snplist[i]["index"])
    h2_last = h2
    for k in range(-para["burn_in"], para["num_iter"]):
        Mc = 0
        h2 = min(h2, 1)
        h2 = (1 - para["prop"]) * h2_last + para["prop"] * h2
        h2_last = h2
        print("Run pmldpred_auto step:", k, "p:", p, "h2:", h2)
        h2_per_var = h2 / (M * p)
        inv_odd_p = (1 - p) / p
        subinput = []
        for i in range(len(PM)):
            subinput.append(
                (
                    PM[i],
                    snplist[i],
                    beta_hat[i],
                    curr_beta[i],
                    R_curr_beta[i],
                    N[i],
                    h2_per_var,
                    inv_odd_p,
                    h2,
                    para,
                    i,
                    k,
                )
            )
        results = Parallel(n_jobs=para["n_jobs"])(
            delayed(pmpred_auto_subprocess)(d) for d in subinput
        )
        h2 = 0
        for i in range(len(PM)):
            curr_beta[i], R_curr_beta[i], Mc_add, h2_add, mean_beta = results[i]
            Mc += Mc_add
            h2 += h2_add
            if k >= 0:
                avg_beta[i] += mean_beta
        p = np.random.beta(1 + Mc, 1 + M - Mc)
    for i in range(len(PM)):
        beta_pmldpred_auto.append(avg_beta[i] / para["num_iter"])
        beta_pmldpred_auto[i] = beta_pmldpred_auto[i] * scale_size[i]
        snplist[i]["index"] = snplist[i]["index"].tolist()
    outpara = {
        "p": p,
        "h2": h2,
        "burn_in": para["burn_in"],
        "num_iter": para["num_iter"],
        "log10rtol": np.log(para["rtol"]) / np.log(10),
        "log10taylor_rtol": np.log(para["taylor_rtol"]) / np.log(10),
        "prop": para["prop"],
    }
    return beta_pmldpred_auto, outpara
