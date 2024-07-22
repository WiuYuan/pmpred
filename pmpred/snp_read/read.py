# read precision matrix

import os
from scipy.sparse import csr_matrix
import pandas_plink as ppl


def snplist_read(snplist_folder_path):
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


def PM_read(precision_folder_path):
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
        PM_block["filename"] = PM_file
        PM.append(PM_block)
        print("Read Precision matrix file:", PM_file)
    return PM


def bed_read(bed_path):
    (bim, fam, bed) = ppl.read_plink(bed_path)
    return {"bed": bed, "bim": bim, "fam": fam}


def sumstats_read(sumstats_path, split, head_name):
    sumstats = {}
    with open(sumstats_path, "r") as file:
        tot = 0
        for line in file:
            line_list = line.strip().split(split)
            if tot == 0:
                title_list = line_list
                for i in range(len(title_list)):
                    sumstats[title_list[i]] = []
            else:
                for i in range(len(title_list)):
                    sumstats[title_list[i]].append(line_list[i])
            tot = tot + 1
            if tot % 100000 == 0:
                print("Read sumstats line:", tot)
    for key in head_name.keys():
        if head_name[key] in sumstats:
            sumstats[key] = sumstats.pop(head_name[key])
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
