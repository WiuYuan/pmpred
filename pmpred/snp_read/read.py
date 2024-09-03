# read precision matrix

import os
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import pandas_plink as ppl
import pandas as pd
import numpy as np


def esnplist_read(snplist_folder_path):
    print("Read snplist from", snplist_folder_path)
    snplist = []
    for snplist_file in sorted(
        os.listdir(snplist_folder_path),
        key=lambda x: (
            int(x.split("_")[1][3:]),
            int(x.split("_")[2]),
            int(x.split("_")[3].split(".")[0]),
        ),
    ):
        snplist_block = {}
        with open(snplist_folder_path + "/" + snplist_file, "r") as file:
            tot = 0
            title_list = []
            index_dict = {}
            for line in file:
                line_list = line.strip().split(",")
                if tot > 0 and line_list[0] in index_dict:
                    continue
                if tot == 0:
                    title_list = line_list
                    for i in range(len(title_list)):
                        snplist_block[title_list[i]] = []
                else:
                    for i in range(len(title_list)):
                        snplist_block[title_list[i]].append(line_list[i])
                    index_dict[snplist_block["index"][tot - 1]] = tot - 1
                tot = tot + 1
        snplist_block["filename"] = snplist_file
        snplist_block["rsid"] = snplist_block.pop("site_ids")
        snplist_block["REF"] = snplist_block.pop("anc_alleles")
        snplist_block["ALT"] = snplist_block.pop("deriv_alleles")
        snplist.append(snplist_block)
        print("Read snplist file:", snplist_file)
    return snplist


def snplist_read(snplist_folder_path):
    print("Read snplist from", snplist_folder_path)
    snplist = []
    for snplist_file in sorted(
        os.listdir(snplist_folder_path),
        key=lambda x: (int(x.split("_")[1].split(".")[0][3:])),
        # key=lambda x: (
        #     int(x.split("_")[1][3:]),
        #     int(x.split("_")[2]),
        #     int(x.split("_")[3].split(".")[0]),
        # ),
    ):
        snplist_block = {}
        snplist_pd = pd.read_csv(snplist_folder_path + "/" + snplist_file, sep="\t")
        for title in snplist_pd.columns:
            snplist_block[title] = np.array(snplist_pd[title]).tolist()
        snplist_block["index"] = np.arange(snplist_pd.shape[0]).tolist()
        # with open(snplist_folder_path + "/" + snplist_file, "r") as file:
        # tot = 0
        # title_list = []
        # index_dict = {}
        # for line in file:
        #     line_list = line.strip().split(",")
        #     if tot > 0 and line_list[0] in index_dict:
        #         continue
        #     if tot == 0:
        #         title_list = line_list
        #         for i in range(len(title_list)):
        #             snplist_block[title_list[i]] = []
        #     else:
        #         for i in range(len(title_list)):
        #             snplist_block[title_list[i]].append(line_list[i])
        #         index_dict[snplist_block["index"][tot - 1]] = tot - 1
        #     tot = tot + 1
        snplist_block["filename"] = snplist_file
        # snplist_block["rsid"] = snplist_block.pop("site_ids")
        # snplist_block["REF"] = snplist_block.pop("anc_alleles")
        # snplist_block["ALT"] = snplist_block.pop("deriv_alleles")
        snplist.append(snplist_block)
        print("Read snplist file:", snplist_file)
    return snplist


def symmetrize_sparse_matrix(A):
    """
    Modify the sparse matrix A so that for each element A[i, j],
    if |A[i, j]| > |A[j, i]|, then A[i, j] = A[j, i].

    Parameters:
    A (scipy.sparse.csr_matrix or scipy.sparse.coo_matrix): The input sparse matrix.

    Returns:
    scipy.sparse.coo_matrix: The symmetrized sparse matrix in COO format.
    """
    if not isinstance(A, (sp.csr_matrix, sp.coo_matrix)):
        raise TypeError(
            "Input must be a scipy.sparse.csr_matrix or scipy.sparse.coo_matrix"
        )

    # Convert to COO format if not already
    if isinstance(A, csr_matrix):
        A = A.tocoo()

    data = A.data
    row = A.row
    col = A.col

    # Dictionary to store the maximum values
    values = {(r, c): 0 for r, c in zip(row, col)}

    # Populate the dictionary with the maximum values
    for r, c, v in zip(row, col, data):
        if abs(v) > abs(values.get((c, r), 0)):
            values[(r, c)] = v
            values[(c, r)] = v

    # Create a new matrix from the dictionary
    new_data = []
    new_row = []
    new_col = []
    for (r, c), v in values.items():
        if v != 0:
            new_data.append(v)
            new_row.append(r)
            new_col.append(c)

    # Convert lists to numpy arrays
    new_data = np.array(new_data)
    new_row = np.array(new_row)
    new_col = np.array(new_col)

    # Create the updated sparse matrix in COO format
    A_updated = sp.csr_matrix((new_data, (new_row, new_col)), shape=A.shape)

    return A_updated


def ePM_read(precision_folder_path):
    print("Read precision matrix from", precision_folder_path)
    PM = []
    for PM_file in sorted(
        os.listdir(precision_folder_path),
        key=lambda x: (
            int(x.split("_")[1][3:]),
            int(x.split("_")[2]),
            int(x.split("_")[3].split(".")[0]),
        ),
    ):
        PM_block = {}
        rows = []
        cols = []
        data = []
        with open(precision_folder_path + "/" + PM_file, "r") as file:
            for line in file:
                row_idx, col_idx, value = map(float, line.strip().split(","))
                rows.append(int(row_idx))
                cols.append(int(col_idx))
                data.append(value)
        PM_block["precision"] = csr_matrix((data, (rows, cols)))
        D = 1 / np.sqrt(np.maximum(PM_block["precision"].diagonal(), 1e-7))
        PM_block["precision"] = (
            PM_block["precision"].multiply(D.reshape(1, -1)).multiply(D.reshape(-1, 1))
        )
        PM_block["precision"] = PM_block["precision"].tocsr()
        PM_block["filename"] = PM_file
        PM.append(PM_block)
        print("Read Precision matrix file:", PM_file)
    return PM


def PM_read(precision_folder_path):
    print("Read precision matrix from", precision_folder_path)
    PM = []
    for PM_file in sorted(
        os.listdir(precision_folder_path),
        key=lambda x: (int(x.split("_")[1].split(".")[0][3:])),
        # key=lambda x: (
        #     int(x.split("_")[1][3:]),
        #     int(x.split("_")[2]),
        #     int(x.split("_")[3].split(".")[0]),
        # ),
    ):
        PM_block = {}
        # rows = []
        # cols = []
        # data = []
        edgelist_pd = pd.read_csv(precision_folder_path + "/" + PM_file, sep="\t")
        # with open(precision_folder_path + "/" + PM_file, "r") as file:
        #     for line in file:
        #         row_idx, col_idx, value = map(float, line.strip().split(","))
        #         rows.append(int(row_idx))
        #         cols.append(int(col_idx))
        #         data.append(value)
        PM_block["precision"] = csr_matrix(
            (edgelist_pd["weight"], (edgelist_pd["from"], edgelist_pd["to"]))
        )
        # PM_block["precision"] = symmetrize_sparse_matrix(PM_block["precision"])
        # PM_block["precision"] += PM_block["precision"].transpose() - sp.diags(
        #     PM_block["precision"].diagonal()
        # )
        PM_block["filename"] = PM_file
        PM.append(PM_block)
        print("Read Precision matrix file:", PM_file)
    return PM


def bed_read(bed_path):
    (bim, fam, bed) = ppl.read_plink(bed_path)
    return {"bed": bed, "bim": bim, "fam": fam}


def sumstats_read(sumstats_path, split, head_name):
    print("Read sumstats from", sumstats_path)
    sumstats = {}
    sumstats_pd = pd.read_csv(sumstats_path, sep=split)
    # with open(sumstats_path, "r") as file:
    #     tot = 0
    #     for line in file:
    #         line_list = line.strip().split(split)
    #         if tot == 0:
    #             title_list = line_list
    #             for i in range(len(title_list)):
    #                 sumstats[title_list[i]] = []
    #         else:
    #             for i in range(len(title_list)):
    #                 sumstats[title_list[i]].append(line_list[i])
    #         tot = tot + 1
    #         if tot % 100000 == 0:
    #             print("Read sumstats line:", tot)
    if len(head_name) == 0:
        for key in sumstats_pd.columns:
            sumstats[key] = np.array(sumstats_pd[key]).tolist()
    else:
        for key in head_name.keys():
            if head_name[key] in sumstats_pd.columns:
                sumstats[key] = np.array(sumstats_pd[head_name[key]]).tolist()
                # sumstats[key] = sumstats.pop(head_name[key])
    return sumstats


def vcf_read(vcf_path):
    vcfstats = {}
    with open(vcf_path, "r") as file:
        tot = 0
        for line in file:
            if line[0:2] == "##":
                continue
            line_list = line.strip().split("\t")
            if tot == 0:
                title_list = line_list
                for i in range(len(title_list)):
                    vcfstats[title_list[i]] = []
            else:
                for i in range(len(title_list)):
                    if title_list[i] == "ID" or title_list[i] == "REF":
                        vcfstats[title_list[i]].append(line_list[i])
                    elif len(line_list[i]) == 3 and line_list[i][1] == "/":
                        v = line_list[i]
                        if v[0] == ".":
                            a = 0
                        else:
                            a = int(v[0])
                        if v[2] == ".":
                            b = 0
                        else:
                            b = int(v[2])
                        vcfstats[title_list[i]].append(a + b)
            tot = tot + 1
            if tot % 1000 == 0:
                print("Read vcf file line:", tot)
    for key in title_list:
        if len(vcfstats[key]) == 0:
            vcfstats.pop(key)
    vcfstats["rsid"] = vcfstats.pop("ID")
    return vcfstats


def fam_read(fam_path):
    famstats = {}
    title_list = [
        "IndividualID",
        "Phenotype",
    ]
    for i in range(len(title_list)):
        famstats[title_list[i]] = []
    with open(fam_path, "r") as file:
        for line in file:
            line_list = line.strip().split(" ")
            famstats["IndividualID"].append(line_list[0] + "_" + line_list[1])
            famstats["Phenotype"].append(line_list[2])
    return famstats


def beta_read(beta_path):
    betastats = {"rsid": [], "REF": [], "beta": []}
    with open(beta_path, "r") as file:
        tot = 0
        for line in file:
            line_list = line.strip().split("\t")
            if tot > 0:
                betastats["rsid"].append(line_list[0])
                betastats["REF"].append(line_list[1])
                betastats["beta"].append(line_list[2])
            tot += 1
    return betastats
