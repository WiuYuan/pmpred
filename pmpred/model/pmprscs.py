import numpy as np
from joblib import Parallel, delayed

# from scipy.sparse.linalg import lcg, cg
import math


def psi(x, alpha, lam):
    f = -alpha * (math.cosh(x) - 1.0) - lam * (math.exp(x) - x - 1.0)
    return f


def dpsi(x, alpha, lam):
    f = -alpha * math.sinh(x) - lam * (math.exp(x) - 1.0)
    return f


def g(x, sd, td, f1, f2):
    if (x >= -sd) and (x <= td):
        f = 1.0
    elif x > td:
        f = f1
    elif x < -sd:
        f = f2

    return f


def gigrnd(p, a, b):
    # setup -- sample from the two-parameter version gig(lam,omega)
    p = float(p)
    a = float(a)
    b = float(b)
    lam = p
    omega = math.sqrt(a * b)

    if lam < 0:
        lam = -lam
        swap = True
    else:
        swap = False

    alpha = math.sqrt(math.pow(omega, 2) + math.pow(lam, 2)) - lam

    # find t
    x = -psi(1.0, alpha, lam)
    if (x >= 0.5) and (x <= 2.0):
        t = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            t = math.sqrt(2.0 / (alpha + lam))
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            t = 1.0
        else:
            t = math.log(4.0 / (alpha + 2.0 * lam))

    # find s
    x = -psi(-1.0, alpha, lam)
    if (x >= 0.5) and (x <= 2.0):
        s = 1.0
    elif x > 2.0:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        else:
            s = math.sqrt(4.0 / (alpha * math.cosh(1) + lam))
    elif x < 0.5:
        if (alpha == 0) and (lam == 0):
            s = 1.0
        elif alpha == 0:
            s = 1.0 / lam
        elif lam == 0:
            s = math.log(
                1.0 + 1.0 / alpha + math.sqrt(1.0 / math.pow(alpha, 2) + 2.0 / alpha)
            )
        else:
            s = min(
                1.0 / lam,
                math.log(
                    1.0
                    + 1.0 / alpha
                    + math.sqrt(1.0 / math.pow(alpha, 2) + 2.0 / alpha)
                ),
            )

    # find auxiliary parameters
    eta = -psi(t, alpha, lam)
    zeta = -dpsi(t, alpha, lam)
    theta = -psi(-s, alpha, lam)
    xi = dpsi(-s, alpha, lam)

    p = 1.0 / xi
    r = 1.0 / zeta

    td = t - r * eta
    sd = s - p * theta
    q = td + sd

    # random variate generation
    while True:
        U = np.random.random()
        V = np.random.random()
        W = np.random.random()
        if U < q / (p + q + r):
            rnd = -sd + q * V
        elif U < (q + r) / (p + q + r):
            rnd = td - r * math.log(V)
        else:
            rnd = -sd + p * math.log(V)

        f1 = math.exp(-eta - zeta * (rnd - t))
        f2 = math.exp(-theta + xi * (rnd + s))
        if W * g(rnd, sd, td, f1, f2) <= math.exp(psi(rnd, alpha, lam)):
            break

    # transform back to the three-parameter version gig(p,a,b)
    rnd = math.exp(rnd) * (
        lam / omega + math.sqrt(1.0 + math.pow(lam, 2) / math.pow(omega, 2))
    )
    if swap:
        rnd = 1.0 / rnd

    rnd = rnd / math.sqrt(a / b)
    return rnd


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


def pmprscs_Q_times(P, x, psi, Pid, para):
    if len(Pid) == 0:
        return np.array([])
    y = np.zeros(P.shape[0])
    y[Pid] = x * np.sqrt(psi)
    return np.sqrt(psi) * cg(P, y, rtol=para["rtol"])[Pid]


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
    return curr_beta * np.sqrt(psi) * np.sqrt(sigma2) / np.sqrt(np.sqrt(N)) + mu


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
    # khi = N / sigma2 * curr_beta**2
    psi = np.zeros(len(N))
    for i in range(len(N)):
        psi[i] = gigrnd(a - 0.5, 2.0 * delta[i], N[i] * curr_beta[i] ** 2 / sigma2)
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
            subinput.append((N[i], sigma2, curr_beta[i], a, delta[i]))
        psi = Parallel(n_jobs=para["n_jobs"])(
            delayed(pmprscs_auto_subprocess_psi)(d) for d in subinput
        )
        subinput = []
        for i in range(len(PM)):
            subinput.append((a, b, psi[i], phi))
        delta = Parallel(n_jobs=para["n_jobs"])(
            delayed(pmprscs_auto_subprocess_delta)(d) for d in subinput
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
