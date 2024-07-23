import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import norm
import pmpred as pm


def generate_sumstats_beta_from_PM(PM, snplist, para):
    sumstats = []
    beta_true = []
    m = 0
    for i in range(len(snplist)):
        m += len(snplist[i]["rsid"])
    si = 0
    sigma = para["h2"] / (m * para["p"])
    for i in range(len(snplist)):
        sumstats_block = {}
        mm = para["sumstats_block_size"]
        sumstats_block["N"] = np.ones(mm) * para["N"]
        sumstats_block["beta_se"] = np.ones(mm) / np.sqrt(para["N"])
        sumstats_block["rsid"] = np.array(snplist[i]["rsid"])[0:mm].tolist()
        si += para["PM_size"]
        beta_true.append([])
        if mm == 0:
            sumstats_block["beta"] = []
            sumstats.append(sumstats_block)
            continue
        for j in range(mm):
            if np.random.rand() < para["p"]:
                beta_true[i].append(np.random.normal(0, np.sqrt(sigma)))
            else:
                beta_true[i].append(0)
        beta_true[i] = np.array(beta_true[i])
        R = PM[i]["LD"]
        sumstats_block["beta"] = np.random.multivariate_normal(
            R @ beta_true[i], R * (1 - para["h2"]) / para["N"]
        ).tolist()
        sumstats.append(sumstats_block)

    return sumstats, beta_true


def generate_sumstats_beta_from_bedstats(bedstats, snplist, para):
    sumstats = {}
    print("Get the genotype matrix")
    genotype = bedstats["bed"].compute()
    geno_rsid = bedstats["bim"]["snp"].tolist()
    geno_rsid_dict = {value: index for index, value in enumerate(geno_rsid)}
    snplist_rsid = []
    bed_rsid = []
    for i in range(len(snplist)):
        rsid1 = [
            index
            for index, value in enumerate(snplist[i]["rsid"])
            if value in geno_rsid_dict
        ]
        snplist_rsid += rsid1
        rsid2 = [
            geno_rsid_dict[value]
            for index, value in enumerate(snplist[i]["rsid"])
            if value in geno_rsid_dict
        ]
        bed_rsid += rsid2
        if i % 137 == 0:
            print("generate_sumstats_beta_from_bedstats block:", i)
    M = len(snplist_rsid)
    sigma = para["h2"] / (M * para["p"])
    z = (np.random.rand(M) < para["p"]).astype(int)
    beta_true_total = z * np.random.normal(0, sigma, M)
    R = genotype[bed_rsid, :]
    nan_columns = np.any(np.isnan(R), axis=0)
    R = R[:, ~nan_columns]
    row_means = np.mean(R, axis=1, keepdims=True)
    R = R - row_means
    row_stds = np.sqrt(np.sum(np.square(R), axis=1, keepdims=True))
    R = R / row_stds
    sumstats["beta"] = np.random.normal(size=M)
    mean = R @ (R.T @ sumstats["beta"])
    sumstats["beta"] = (
        mean + np.sqrt((1 - para["h2"]) / para["N"]) * R.T @ sumstats["beta"]
    ).tolist()
    sumstats["N"] = np.ones(M) * para["N"]
    sumstats["beta_se"] = np.ones(M)
    sumstats["rsid"] = bedstats["bim"]["snp"][bed_rsid].tolist()
    sumstats["REF"] = bedstats["bim"]["a0"][bed_rsid].tolist()
    sumstats["ALT"] = bedstats["bim"]["a1"][bed_rsid].tolist()
    return sumstats, beta_true_total


def generate_phenotype(phestats, beta, para):
    phestats["Phenotype"] = phestats["X"] @ beta + np.random.randn(
        len(phestats["Phenotype"])
    ) * np.sqrt(1 - para["h2"])


def generate_PM_snplist(para):
    snplist = []
    PM = []
    si = 0
    for i in range(para["block_num"]):
        snplist_block = {}
        PM_block = {}
        N = para["PM_size"]
        snplist_block["rsid"] = [
            "rs{}".format(i) for i in range(si, si + para["sumstats_block_size"])
        ]
        si += N
        snplist_block["index"] = np.arange(para["sumstats_block_size"])
        M = np.random.rand(N, N) - 0.5
        M += M.T
        R = np.dot(M, M.T)
        D = np.sqrt(np.diag(R))
        PM_block["LD"] = R / np.outer(D, D)
        PM_block["precision"] = csr_matrix(np.linalg.inv(PM_block["LD"]))
        PM_block["LD"] = PM_block["LD"][
            0 : para["sumstats_block_size"], 0 : para["sumstats_block_size"]
        ]
        snplist.append(snplist_block)
        PM.append(PM_block)
    return PM, snplist


def generate_beta_and_se_from_p_and_z(sumstats):
    print("Generate beta and beta_se from p value and z scores in sumstats")
    for i in range(len(sumstats)):
        sumstats[i]["beta"] = (
            np.abs(norm.ppf(np.array(sumstats[i]["p"]).astype(float) / 2))
            * np.sign(np.array(sumstats[i]["z"]).astype(float))
        ).tolist()
        sumstats[i]["beta_se"] = np.ones(len(sumstats[i]["beta"])).tolist()


def generate_p_and_z_from_beta_and_se(sumstats):
    print("Generate p value and z scores from beta and beta_se in sumstats")
    sumstats["z"] = (
        np.array(sumstats["beta"]) / np.array(sumstats["beta_se"])
    ).tolist()
    sumstats["p"] = (2 * (1 - norm.cdf(np.abs(np.array(sumstats["z"]))))).tolist()


def generate_N_in_sumstats_list(sumstats_list, N):
    print("Generate N in sumstats_list")
    sumstats_list["N"] = (np.ones(len(sumstats_list["rsid"])) * N).tolist()


def get_para():
    para = {}
    para["h2"] = 0.5
    para["p"] = 0.18
    para["sumstats_block_size"] = 10
    para["PM_size"] = 20
    para["block_num"] = 10
    para["burn_in"] = 50
    para["num_iter"] = 100
    para["N"] = 1000
    para["prop"] = 1
    para["taylor_rtol"] = 0.01
    para["h2_min"] = 0
    para["h2_max"] = 1
    para["p_min"] = 0
    para["p_max"] = 1
    para["prop"] = 0.1
    para["n_jobs"] = -1
    para["rtol"] = 1e-10
    para["prscs_a"] = 1.5
    para["prscs_b"] = 0.5
    para["output_key"] = ["rsid", "REF", "beta_joint"]
    return para
