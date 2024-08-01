import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed


def filter_by_PM(PM, snplist):
    for i in range(len(PM)):
        PMid = np.where(np.diff(PM[i]["precision"].indptr) != 0)[0]
        PM[i]["precision"] = PM[i]["precision"][PMid][:, PMid]
        for key in list(snplist[i].keys()):
            if isinstance(snplist[i][key], list):
                snplist[i][key] = np.array(snplist[i][key])[PMid].tolist()
        if i % 137 == 0:
            print("Filter_by_PM block:", i)
        snplist[i]["index"] = np.arange(len(snplist[i]["rsid"])).tolist()


def filter_sumstats(sumstats):
    base_map = {"A": "T", "T": "A", "C": "G", "G": "C"}
    N_mean = np.sum(np.array(sumstats["N"]).astype(float)) / len(sumstats["N"])
    i = 0
    while i < len(sumstats["N"]):
        if (
            int(sumstats["N"][i]) < N_mean / 3
            # and sumstats["REF"][i] in base_map.keys()
            # and sumstats["ALT"][i] in base_map.keys()
            and base_map[sumstats["REF"][i]] == sumstats["ALT"][i]
        ):
            for key in sumstats.keys():
                del sumstats[key][i]
        else:
            i += 1
        if i % 100000 == 0:
            print("Filter sumstats line:", i)


def normalize_PM_subprocess(subinput):
    P, i = subinput
    print("normalize_PM_subprocess block:", i)
    Pid = sorted(set(P.tocoo().row))
    D = np.zeros(P.shape[0])
    D[Pid] = np.sqrt(np.linalg.inv(P[Pid][:, Pid].toarray()).diagonal())
    return P.multiply(np.outer(D, D)).tocsr()


def normalize_PM_parallel(PM, para):
    subinput = []
    for i in range(len(PM)):
        subinput.append((PM[i]["precision"], i))
    results = Parallel(n_jobs=para["n_jobs"])(
        delayed(normalize_PM_subprocess)(d) for d in subinput
    )
    for i in range(len(PM)):
        PM[i]["precision"] = results[i]


def enormalize_PM_subprocess(subinput):
    P, i = subinput
    P = sp.triu(P).tocsr()
    print("enormalize_PM_subprocess block:", i)
    Pid = sorted(set(P.tocoo().row))
    D = np.zeros(P.shape[0])
    D[Pid] = np.sqrt(np.linalg.inv(P[Pid][:, Pid].toarray()).diagonal())
    return P.multiply(np.outer(D, D)).tocsr()


def enormalize_PM_parallel(PM, para):
    subinput = []
    for i in range(len(PM)):
        subinput.append((PM[i]["precision"], i))
    results = Parallel(n_jobs=para["n_jobs"])(
        delayed(enormalize_PM_subprocess)(d) for d in subinput
    )
    for i in range(len(PM)):
        PM[i]["precision"] = results[i]


def normalize_dense_matrix(A):
    diag_elements = np.diag(A)
    sqrt_diag_outer = np.sqrt(np.outer(diag_elements, diag_elements))
    return A / sqrt_diag_outer


def filter_by_sumstats_subprocess(subinput):
    rsid_sumstats, snplist_rsid, i = subinput
    rsid1 = [
        index for index, value in enumerate(snplist_rsid) if value in rsid_sumstats
    ]
    rsid2 = [
        rsid_sumstats[value]
        for index, value in enumerate(snplist_rsid)
        if value in rsid_sumstats
    ]
    if i % 137 == 0:
        print("Filter_by_sumstats parallel block:", i)
    return rsid1, rsid2


def filter_by_sumstats_parallel(PM, snplist, sumstats, para):
    sumstats_set = []
    rsid_sumstats = {value: index for index, value in enumerate(sumstats["rsid"])}
    subinput = []
    for i in range(len(PM)):
        subinput.append((rsid_sumstats, snplist[i]["rsid"], i))
    for key in list(sumstats.keys()):
        if isinstance(sumstats[key], list):
            sumstats[key] = np.array(sumstats[key])
    results = Parallel(n_jobs=para["n_jobs"], backend="threading")(
        delayed(filter_by_sumstats_subprocess)(d) for d in subinput
    )
    for i in range(len(PM)):
        rsid1, rsid2 = results[i]
        for key in list(snplist[i].keys()):
            if isinstance(snplist[i][key], list):
                snplist[i][key] = np.array(snplist[i][key])[rsid1].tolist()
        sumstats_block = {}
        for key in list(sumstats.keys()):
            if isinstance(sumstats[key], np.ndarray):
                sumstats_block[key] = sumstats[key][rsid2].tolist()
        sumstats_set.append(sumstats_block)
        if i % 137 == 0:
            print("Filter_by_sumstats results block:", i)
    return sumstats_set


def filter_by_unique_sumstats(sumstats):
    print("Make sumstats rsid unique")
    unique_indices = set()
    unique_indices = [
        unique_indices.add(x) or i
        for i, x in enumerate(sumstats["rsid"])
        if x not in unique_indices
    ]
    for key in sumstats.keys():
        if isinstance(sumstats[key], list):
            sumstats[key] = np.array(sumstats[key])[unique_indices].tolist()


def filter_by_unique_snplist(snplist):
    unique_indices = set()
    for i in range(len(snplist)):
        unique_block_indices = [
            unique_indices.add(x) or i
            for i, x in enumerate(snplist[i]["rsid"])
            if x not in unique_indices
        ]
        for key in snplist[i].keys():
            if isinstance(snplist[i][key], list):
                snplist[i][key] = np.array(snplist[i][key])[
                    unique_block_indices
                ].tolist()
        if i % 137 == 0:
            print("Make snplist unique block:", i)


def filter_by_sumstats(PM, snplist, sumstats):
    sumstats_set = []
    rsid_sumstats = {value: index for index, value in enumerate(sumstats["rsid"])}
    for i in range(len(PM)):
        rsid = [
            index
            for index, value in enumerate(snplist[i]["rsid"])
            if value in rsid_sumstats
        ]
        for key in list(snplist[i].keys()):
            if isinstance(snplist[i][key], list):
                snplist[i][key] = np.array(snplist[i][key])[rsid].tolist()
        rsid = [
            rsid_sumstats[value]
            for index, value in enumerate(snplist[i]["rsid"])
            if value in rsid_sumstats
        ]
        sumstats_block = {}
        for key in list(sumstats.keys()):
            if isinstance(sumstats[key], list):
                sumstats_block[key] = np.array(sumstats[key])[rsid].tolist()
        sumstats_set.append(sumstats_block)
        print("Filter_by_sumstats block:", i)
    return sumstats_set


def merge_vcf_fam(vcfstats, famstats):
    phestats = {}
    N = len(famstats["IndividualID"])
    M = len(vcfstats["rsid"])
    X = np.zeros((N, M))
    for i in range(N):
        if i % 1000 == 0:
            print("merge_vcf_fam Individual:", i)
        for j in range(M):
            X[i][j] = vcfstats[famstats["IndividualID"][i]][j]
    phestats["X"] = X
    phestats["rsid"] = vcfstats["rsid"][:]
    phestats["REF"] = vcfstats["REF"][:]
    # phestats["ALT"] = vcfstats["ALT"][:]
    phestats["Phenotype"] = np.array(famstats["Phenotype"])
    del vcfstats
    return phestats


def filter_by_vcf(betastats, phestats):
    rsid_phestats = {value: index for index, value in enumerate(phestats["rsid"])}
    rsid = [
        index for index, value in enumerate(betastats["rsid"]) if value in rsid_phestats
    ]
    for key in list(betastats.keys()):
        if isinstance(betastats[key], list):
            betastats[key] = np.array(betastats[key])[rsid].tolist()
    rsid_betastats = {value: index for index, value in enumerate(betastats["rsid"])}
    rsid = [
        index for index, value in enumerate(phestats["rsid"]) if value in rsid_betastats
    ]
    for key in list(phestats.keys()):
        if isinstance(phestats[key], list):
            phestats[key] = np.array(phestats[key])[rsid].tolist()
    phestats["X"] = phestats["X"][:, rsid]


def merge_beta(beta):
    beta_total = []
    for i in range(len(beta)):
        if isinstance(beta[i], list) == 0:
            beta_total += beta[i].tolist()
        else:
            beta_total += beta[i]
    return np.array(beta_total)


def filter_by_REF_ALT(betastats, phestats):
    for j in range(len(phestats["rsid"])):
        if betastats["REF"][j] != phestats["REF"][j]:
            phestats["X"][:, j] = 2 - phestats["X"][:, j]


def PM_get_LD_subprocess(subinput):
    P, Pid, i = subinput
    # Pid = sorted(set(P.tocoo().row))
    R = sp.linalg.inv(P.tocsc())[Pid][:, Pid].toarray()
    D = np.sqrt(R.diagonal())
    print("PM_get_LD_subprocess block:", i)
    return R / np.outer(D, D)


def PM_get_LD(PM, snplist, para):
    subinput = []
    for i in range(len(PM)):
        subinput.append((PM[i]["precision"], snplist[i]["index"], i))
    results = Parallel(n_jobs=para["n_jobs"])(
        delayed(PM_get_LD_subprocess)(d) for d in subinput
    )
    for i in range(len(PM)):
        PM[i]["LD"] = results[i]
