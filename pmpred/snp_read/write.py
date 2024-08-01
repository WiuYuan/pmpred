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
                            f.write(f"{sumstats[i][key][j]:.11f}\t")
                    else:
                        f.write("NA\t")
                f.write(f"{beta[i][j]:.11f}")
                f.write("\n")


def sumstats_write(sumstats, split, out_head_name, output_path):
    with open(output_path, "w") as f:
        for j in range(len(list(out_head_name.keys()))):
            key = list(out_head_name.keys())[j]
            f.write(out_head_name[key])
            if j != len(list(out_head_name.keys())) - 1:
                f.write(split)
        f.write("\n")
        for i in range(len(sumstats["rsid"])):
            for j in range(len(list(out_head_name.keys()))):
                key = list(out_head_name.keys())[j]
                if isinstance(sumstats[key][i], str):
                    f.write(sumstats[key][i])
                elif key == "N":
                    f.write(f"{int(sumstats[key][i])}")
                else:
                    f.write(f"{sumstats[key][i]:.11f}")
                if j != len(list(out_head_name.keys())) - 1:
                    f.write(split)
            f.write("\n")


def sumstats_set_write(sumstats_set, split, out_head_name, output_path):
    with open(output_path, "w") as f:
        for j in range(len(list(out_head_name.keys()))):
            key = list(out_head_name.keys())[j]
            f.write(out_head_name[key])
            if j != len(list(out_head_name.keys())) - 1:
                f.write(split)
        f.write("\n")
        for k in range(len(sumstats_set)):
            sumstats = sumstats_set[k]
            for i in range(len(sumstats["rsid"])):
                for j in range(len(list(out_head_name.keys()))):
                    key = list(out_head_name.keys())[j]
                    if isinstance(sumstats[key][i], str):
                        f.write(sumstats[key][i])
                    elif isinstance(sumstats[key][i], int):
                        f.write(f"{int(sumstats[key][i])}")
                    else:
                        f.write(f"{sumstats[key][i]:.11f}")
                    if j != len(list(out_head_name.keys())) - 1:
                        f.write(split)
                f.write("\n")


def phestats_write(phestats, split, output_path):
    with open(output_path, "w") as f:
        for i in range(len(phestats["phenotype"])):
            for j in range(len(list(phestats.keys()))):
                key = list(phestats.keys())[j]
                if isinstance(phestats[key][i], str):
                    f.write(phestats[key][i])
                elif isinstance(phestats[key][i], int):
                    f.write(f"{int(phestats[key][i])}")
                else:
                    f.write(f"{phestats[key][i]:.11f}")
                if j != len(list(phestats.keys())) - 1:
                    f.write(split)
            f.write("\n")


def PM_write(PM, output_folder_path):
    for i in range(len(PM)):
        with open(output_folder_path + "/" + PM[i]["filename"], "w") as f:
            Q = PM[i]["precision"].tocoo()
            for (row, col), value in zip(zip(Q.row, Q.col), Q.data):
                f.write(f"{row},{col},{value:.11f}\n")
