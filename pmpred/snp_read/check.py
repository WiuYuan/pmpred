def check_same_rsid(snplist, sumstats):
    for i in range(len(snplist)):
        if snplist[i]["rsid"] != sumstats[i]["rsid"]:
            raise Exception("The rsids of the snplist are not same as sumstats!")


def check_unique_rsid_sumstats(sumstats):
    rsid_set = set()
    for i in range(len(sumstats)):
        rsid_block_set = set(sumstats[i]["rsid"])
        s = len(rsid_set)
        rsid_set.update(rsid_block_set)
        if s + len(sumstats[i]["rsid"]) != len(rsid_set):
            raise Exception("The rsids of sumstats are have same element!")
