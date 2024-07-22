def sumstats_beta_write(sumstats, beta, output_path, run_time, outpara, head_name):
    # output_key = ["rsid", "CHR", "POS", "REF", "ALT", "beta", "beta_se", "P", "N"]
    output_key = ["rsid", "REF"]
    with open(output_path, "w") as f:
        f.write(f"## Running time: {run_time:.4f} seconds\n")
        f.write(f"##")
        for key in outpara.keys():
            f.write(f" {key} = {outpara[key]:.4f}")
        f.write("\n")
        f.write("\t".join([head_name[key] for key in output_key]))
        f.write("\tbeta_joint\n")
        for i in range(len(sumstats)):
            for j in range(len(beta[i])):
                for key in output_key:
                    if key in sumstats[i]:
                        if isinstance(sumstats[i][key][j], str):
                            f.write(sumstats[i][key][j] + "\t")
                        elif key == "N":
                            f.write(f"{int(sumstats[i][key][j])}\t")
                        else:
                            f.write(f"{sumstats[i][key][j]:.6f}\t")
                    else:
                        f.write("NA\t")
                f.write(f"{beta[i][j]:.6f}")
                f.write("\n")


def sumstats_write(sumstats, output_path):
    output_key = ["rsid", "REF", "ALT", "beta_se", "beta"]
    with open(output_path, "w") as f:
        f.write("\t".join([key for key in output_key]))
        f.write("\n")
        for i in range(len(sumstats["rsid"])):
            for key in output_key:
                if key in sumstats:
                    if isinstance(sumstats[key][i], str):
                        f.write(sumstats[key][i] + "\t")
                    elif key == "N":
                        f.write(f"{int(sumstats[key][i])}\t")
                    else:
                        f.write(f"{sumstats[key][i]:.6f}\t")
            f.write("\n")


def PM_write(PM, output_folder_path):
    for i in range(len(PM)):
        with open(output_folder_path + "/" + PM[i]["filename"], "w") as f:
            Q = PM[i]["precision"].tocoo()
            for (row, col), value in zip(zip(Q.row, Q.col), Q.data):
                f.write(f"{row},{col},{value:.6f}\n")
