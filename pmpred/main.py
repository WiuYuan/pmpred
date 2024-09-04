#!/usr/bin/env python3

import argparse
import numpy as np
import pmpred as pm
import scipy.sparse as sp
import time


def main():
    parser = argparse.ArgumentParser(
        description="Using Precision Matrix, Snplists and Sumstats to give a joint effective size"
    )
    parser.add_argument("--pm", type=str, help="Precision Matrix Folder Path")
    parser.add_argument("--snp", type=str, help="Snplists Folder Path")
    parser.add_argument("-s", "--sumstats", type=str, help="Sumstats File")
    parser.add_argument(
        "--rsidname",
        type=str,
        default="rsid",
        help="The header name of rsid. The default is rsid.",
    )
    parser.add_argument(
        "--REFname",
        type=str,
        default="REF",
        help="The header name of REF. The default is REF",
    )
    parser.add_argument(
        "--ALTname",
        type=str,
        default="ALT",
        help="The header name of ALT. The default is ALT",
    )
    parser.add_argument(
        "--betaname",
        type=str,
        default="beta",
        help="The header name of beta. The default is beta",
    )
    parser.add_argument(
        "--sename",
        type=str,
        default="beta_se",
        help="The header name of beta se. The default is beta_se",
    )
    parser.add_argument(
        "--Nname",
        type=str,
        default="N",
        help="The header name of N. The default is N",
    )
    parser.add_argument(
        "--pname",
        type=str,
        default="p",
        help="The header name of p value. The default is p",
    )
    parser.add_argument(
        "--zname",
        type=str,
        default="z",
        help="The header name of z scores. The default is z",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="\t",
        help="The split of sumstats. The default is tap",
    )
    parser.add_argument("-o", "--out", type=str, help="Output File")
    parser.add_argument("--N", type=int, help="Specify N in GWAS sumstats if need")
    parser.add_argument(
        "--burnin",
        type=int,
        default=50,
        help="The number of burn-in iterations used by the Gibbs sampler The default is 50.",
    )
    parser.add_argument(
        "--numiter",
        type=int,
        default=100,
        help="The number of iterations used by the Gibbs sampler. The default is 100.",
    )
    parser.add_argument(
        "--subiter",
        type=int,
        default=100,
        help="The number of iterations used by solving linear equation. The default is 100.",
    )
    parser.add_argument(
        "--taylor",
        type=float,
        default=0.001,
        help="The number of approximation in taylor expansion. The default is 1e-3.",
    )
    parser.add_argument(
        "--h2",
        type=float,
        default=np.random.rand(),
        help="The genome-wide heritability assumed by PMpred. The default is random",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=np.random.rand(),
        help="The prior probability of non-sparse. The default is random",
    )
    parser.add_argument(
        "--njobs",
        type=int,
        default=-1,
        help="The jobs parallelized. The default is -1",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pmldpred_auto",
        help="The method we use in PMpred, contain {pmldpred_auto, pmldpred_grid, pmprscs, ldpred_inf, normalizepm}. The default is pmldpred_auto",
    )
    parser.add_argument(
        "--usepvalue",
        action="store_true",
        help="If select, then will use p value and z scores to generate beta and beta_se",
    )
    parser.add_argument(
        "--useprevread",
        action="store_true",
        help="If select, then will read snplist and precision matrix from ldgm format",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-10,
        help="The precision of gmres algorithm to solve the linear equation. The default is 1e-10",
    )
    parser.add_argument(
        "--prop",
        type=float,
        default=0.1,
        help="The proprotion of the update of h2. The default is 0.1",
    )
    args = parser.parse_args()
    para = pm.generate.get_para()
    para["h2"] = args.h2
    para["p"] = args.p
    para["burn_in"] = args.burnin
    para["num_iter"] = args.numiter
    para["taylor_rtol"] = args.taylor
    para["n_jobs"] = args.njobs
    para["rtol"] = args.rtol
    para["prop"] = args.prop
    para["subiter"] = args.subiter
    head_name = {
        "rsid": args.rsidname,
        "beta": args.betaname,
        "beta_se": args.sename,
        "REF": args.REFname,
        "ALT": args.ALTname,
        "p": args.pname,
        "z": args.zname,
        "N": args.Nname,
    }
    if args.method == "pmldpred_auto":
        if not args.pm:
            parser.error("You must specify a Precision Matrix Folder Path.")
        if not args.snp:
            parser.error("You must specify a Snplists Folder Path.")
        if not args.sumstats:
            parser.error("You must specify a Sumstats File.")
        start_time = time.time()
        sumstats_list = pm.read.sumstats_read(args.sumstats, args.split, head_name)
        if args.N:
            pm.generate.generate_N_in_sumstats_list(sumstats_list, args.N)
        pm.filter.filter_sumstats(sumstats_list)
        pm.filter.filter_by_unique_sumstats(sumstats_list)
        if args.useprevread:
            PM = pm.read.ePM_read(args.pm)
            snplist = pm.read.esnplist_read(args.snp)
        else:
            PM = pm.read.PM_read(args.pm)
            snplist = pm.read.snplist_read(args.snp)
        pm.filter.filter_by_PM(PM, snplist)
        pm.filter.filter_by_unique_snplist(snplist)
        sumstats = pm.filter.filter_by_sumstats_parallel(
            PM, snplist, sumstats_list, para
        )
        if args.usepvalue:
            pm.generate.generate_beta_and_se_from_p_and_z(sumstats)
        pm.check.check_same_rsid(snplist, sumstats)
        beta_pmldpred_auto, outpara = pm.pmldpred.pmldpred_auto(
            PM, snplist, sumstats, para
        )
        end_time = time.time()
        pm.write.sumstats_beta_write(
            sumstats,
            beta_pmldpred_auto,
            args.out,
            end_time - start_time,
            outpara,
            head_name,
        )
    elif args.method == "pmldpred_grid":
        if not args.pm:
            parser.error("You must specify a Precision Matrix Folder Path.")
        if not args.snp:
            parser.error("You must specify a Snplists Folder Path.")
        if not args.sumstats:
            parser.error("You must specify a Sumstats File.")
        start_time = time.time()
        sumstats_list = pm.read.sumstats_read(args.sumstats, args.split, head_name)
        if args.N:
            pm.generate.generate_N_in_sumstats_list(sumstats_list, args.N)
        pm.filter.filter_sumstats(sumstats_list)
        pm.filter.filter_by_unique_sumstats(sumstats_list)
        if args.useprevread:
            PM = pm.read.ePM_read(args.pm)
            snplist = pm.read.esnplist_read(args.snp)
        else:
            PM = pm.read.PM_read(args.pm)
            snplist = pm.read.snplist_read(args.snp)
        pm.filter.filter_by_PM(PM, snplist)
        pm.filter.filter_by_unique_snplist(snplist)
        sumstats = pm.filter.filter_by_sumstats_parallel(
            PM, snplist, sumstats_list, para
        )
        if args.usepvalue:
            pm.generate.generate_beta_and_se_from_p_and_z(sumstats)
        pm.check.check_same_rsid(snplist, sumstats)
        beta_pmldpred_grid, outpara = pm.pmldpred.pmldpred_grid(
            PM, snplist, sumstats, para
        )
        end_time = time.time()
        pm.write.sumstats_beta_write(
            sumstats,
            beta_pmldpred_grid,
            args.out,
            end_time - start_time,
            outpara,
            head_name,
        )
    elif args.method == "prscs_auto":
        if not args.pm:
            parser.error("You must specify a Precision Matrix Folder Path.")
        if not args.snp:
            parser.error("You must specify a Snplists Folder Path.")
        if not args.sumstats:
            parser.error("You must specify a Sumstats File.")
        start_time = time.time()
        sumstats_list = pm.read.sumstats_read(args.sumstats, args.split, head_name)
        if args.N:
            pm.generate.generate_N_in_sumstats_list(sumstats_list, args.N)
        pm.filter.filter_sumstats(sumstats_list)
        pm.filter.filter_by_unique_sumstats(sumstats_list)
        if args.useprevread:
            PM = pm.read.ePM_read(args.pm)
            snplist = pm.read.esnplist_read(args.snp)
        else:
            PM = pm.read.PM_read(args.pm)
            snplist = pm.read.snplist_read(args.snp)
        pm.filter.filter_by_PM(PM, snplist)
        pm.filter.filter_by_unique_snplist(snplist)
        sumstats = pm.filter.filter_by_sumstats_parallel(
            PM, snplist, sumstats_list, para
        )
        if args.usepvalue:
            pm.generate.generate_beta_and_se_from_p_and_z(sumstats)
        pm.check.check_same_rsid(snplist, sumstats)
        pm.filter.PM_get_LD(PM, snplist, para)
        beta_prscs_auto, outpara = pm.prscs.prscs_auto(PM, snplist, sumstats, para)
        end_time = time.time()
        pm.write.sumstats_beta_write(
            sumstats,
            beta_prscs_auto,
            args.out,
            end_time - start_time,
            outpara,
            head_name,
        )
    elif args.method == "pmprscs_auto":
        if not args.pm:
            parser.error("You must specify a Precision Matrix Folder Path.")
        if not args.snp:
            parser.error("You must specify a Snplists Folder Path.")
        if not args.sumstats:
            parser.error("You must specify a Sumstats File.")
        start_time = time.time()
        sumstats_list = pm.read.sumstats_read(args.sumstats, args.split, head_name)
        if args.N:
            pm.generate.generate_N_in_sumstats_list(sumstats_list, args.N)
        pm.filter.filter_sumstats(sumstats_list)
        pm.filter.filter_by_unique_sumstats(sumstats_list)
        if args.useprevread:
            PM = pm.read.ePM_read(args.pm)
            snplist = pm.read.esnplist_read(args.snp)
        else:
            PM = pm.read.PM_read(args.pm)
            snplist = pm.read.snplist_read(args.snp)
        pm.filter.filter_by_PM(PM, snplist)
        pm.filter.filter_by_unique_snplist(snplist)
        sumstats = pm.filter.filter_by_sumstats_parallel(
            PM, snplist, sumstats_list, para
        )
        if args.usepvalue:
            pm.generate.generate_beta_and_se_from_p_and_z(sumstats)
        pm.check.check_same_rsid(snplist, sumstats)
        beta_pmprscs_auto, outpara = pm.pmprscs.pmprscs_auto(
            PM, snplist, sumstats, para
        )
        end_time = time.time()
        pm.write.sumstats_beta_write(
            sumstats,
            beta_pmprscs_auto,
            args.out,
            end_time - start_time,
            outpara,
            head_name,
        )
    elif args.method == "ldpred_inf":
        if not args.pm:
            parser.error("You must specify a Precision Matrix Folder Path.")
        if not args.snp:
            parser.error("You must specify a Snplists Folder Path.")
        if not args.sumstats:
            parser.error("You must specify a Sumstats File.")
        start_time = time.time()
        sumstats_list = pm.read.sumstats_read(args.sumstats, args.split, head_name)
        if args.N:
            pm.generate.generate_N_in_sumstats_list(sumstats_list, args.N)
        pm.filter.filter_sumstats(sumstats_list)
        pm.filter.filter_by_unique_sumstats(sumstats_list)
        if args.useprevread:
            PM = pm.read.ePM_read(args.pm)
            snplist = pm.read.esnplist_read(args.snp)
        else:
            PM = pm.read.PM_read(args.pm)
            snplist = pm.read.snplist_read(args.snp)
        pm.filter.filter_by_PM(PM, snplist)
        pm.filter.filter_by_unique_snplist(snplist)
        sumstats = pm.filter.filter_by_sumstats_parallel(
            PM, snplist, sumstats_list, para
        )
        if args.usepvalue:
            pm.generate.generate_beta_and_se_from_p_and_z(sumstats)
        pm.check.check_same_rsid(snplist, sumstats)
        beta_ldpred_inf, outpara = pm.ldpred.ldpred_inf(PM, snplist, sumstats, para)
        end_time = time.time()
        pm.write.sumstats_beta_write(
            sumstats,
            beta_ldpred_inf,
            args.out,
            end_time - start_time,
            outpara,
            head_name,
        )
    elif args.method == "ldpred_auto":
        if not args.pm:
            parser.error("You must specify a Precision Matrix Folder Path.")
        if not args.snp:
            parser.error("You must specify a Snplists Folder Path.")
        if not args.sumstats:
            parser.error("You must specify a Sumstats File.")
        start_time = time.time()
        sumstats_list = pm.read.sumstats_read(args.sumstats, args.split, head_name)
        if args.N:
            pm.generate.generate_N_in_sumstats_list(sumstats_list, args.N)
        pm.filter.filter_sumstats(sumstats_list)
        pm.filter.filter_by_unique_sumstats(sumstats_list)
        if args.useprevread:
            PM = pm.read.ePM_read(args.pm)
            snplist = pm.read.esnplist_read(args.snp)
        else:
            PM = pm.read.PM_read(args.pm)
            snplist = pm.read.snplist_read(args.snp)
        pm.filter.filter_by_PM(PM, snplist)
        pm.filter.filter_by_unique_snplist(snplist)
        sumstats = pm.filter.filter_by_sumstats_parallel(
            PM, snplist, sumstats_list, para
        )
        if args.usepvalue:
            pm.generate.generate_beta_and_se_from_p_and_z(sumstats)
        pm.check.check_same_rsid(snplist, sumstats)
        pm.filter.PM_get_LD(PM, snplist, para)
        beta_ldpred_auto, outpara = pm.ldpred.ldpred_auto(PM, snplist, sumstats, para)
        end_time = time.time()
        pm.write.sumstats_beta_write(
            sumstats,
            beta_ldpred_auto,
            args.out,
            end_time - start_time,
            outpara,
            head_name,
        )
    elif args.method == "ldpred_grid":
        if not args.pm:
            parser.error("You must specify a Precision Matrix Folder Path.")
        if not args.snp:
            parser.error("You must specify a Snplists Folder Path.")
        if not args.sumstats:
            parser.error("You must specify a Sumstats File.")
        start_time = time.time()
        sumstats_list = pm.read.sumstats_read(args.sumstats, args.split, head_name)
        if args.N:
            pm.generate.generate_N_in_sumstats_list(sumstats_list, args.N)
        pm.filter.filter_sumstats(sumstats_list)
        pm.filter.filter_by_unique_sumstats(sumstats_list)
        if args.useprevread:
            PM = pm.read.ePM_read(args.pm)
            snplist = pm.read.esnplist_read(args.snp)
        else:
            PM = pm.read.PM_read(args.pm)
            snplist = pm.read.snplist_read(args.snp)
        pm.filter.filter_by_PM(PM, snplist)
        pm.filter.filter_by_unique_snplist(snplist)
        sumstats = pm.filter.filter_by_sumstats_parallel(
            PM, snplist, sumstats_list, para
        )
        if args.usepvalue:
            pm.generate.generate_beta_and_se_from_p_and_z(sumstats)
        pm.check.check_same_rsid(snplist, sumstats)
        pm.filter.PM_get_LD(PM, snplist, para)
        beta_ldpred_grid, outpara = pm.ldpred.ldpred_grid(PM, snplist, sumstats, para)
        end_time = time.time()
        pm.write.sumstats_beta_write(
            sumstats,
            beta_ldpred_grid,
            args.out,
            end_time - start_time,
            outpara,
            head_name,
        )
    elif args.method == "epmldpred_auto":
        if not args.pm:
            parser.error("You must specify a Precision Matrix Folder Path.")
        if not args.snp:
            parser.error("You must specify a Snplists Folder Path.")
        if not args.sumstats:
            parser.error("You must specify a Sumstats File.")
        start_time = time.time()
        sumstats_list = pm.read.sumstats_read(args.sumstats, args.split, head_name)
        if args.N:
            pm.generate.generate_N_in_sumstats_list(sumstats_list, args.N)
        pm.filter.filter_sumstats(sumstats_list)
        pm.filter.filter_by_unique_sumstats(sumstats_list)
        if args.useprevread:
            PM = pm.read.ePM_read(args.pm)
            snplist = pm.read.esnplist_read(args.snp)
        else:
            PM = pm.read.PM_read(args.pm)
            snplist = pm.read.snplist_read(args.snp)
        pm.filter.filter_by_PM(PM, snplist)
        pm.filter.filter_by_unique_snplist(snplist)
        sumstats = pm.filter.filter_by_sumstats_parallel(
            PM, snplist, sumstats_list, para
        )
        if args.usepvalue:
            pm.generate.generate_beta_and_se_from_p_and_z(sumstats)
        pm.check.check_same_rsid(snplist, sumstats)
        beta_epmldpred_auto, outpara = pm.epmldpred.pmldpred_auto(
            PM, snplist, sumstats, para
        )
        end_time = time.time()
        pm.write.sumstats_beta_write(
            sumstats,
            beta_epmldpred_auto,
            args.out,
            end_time - start_time,
            outpara,
            head_name,
        )
    elif args.method == "normalizepm":
        if not args.pm:
            parser.error("You must specify a Precision Matrix Folder Path.")
        PM = pm.read.PM_read(args.pm)
        for i in range(len(PM)):
            P = PM[i]["precision"]
            PM[i]["precision"] = P + P.transpose() - sp.diags(P.diagonal())
        pm.filter.normalize_PM_parallel(PM, para)
        pm.write.PM_write(PM, args.out)
    else:
        parser.error(
            "You must choose --method in {pmldpred_auto, pmldpred_grid, pmprscs_auto, ldpred_inf, normalizepm}"
        )


if __name__ == "__main__":
    main()
