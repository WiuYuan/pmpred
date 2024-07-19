
# PMpred

PMpred is a Python based software package that adjusts GWAS summary statistics
for the effects of precision matrix (PM), which is the inverse of linkage disequilibrium (LD).

* The current version is 1.0.0

## Getting Started

PMpred can be installed using pip on most systems by typing

`pip install pmpred`

### Requirements

LDpred currently requires three Python packages to be installed and in path.  These
are **numpy** [https://numpy.org/](https://numpy.org/), **scipy** [http://www.scipy.org/](http://www.scipy.org/)
and **joblib** [https://joblib.readthedocs.io/en/stable/](https://joblib.readthedocs.io/en/stable/).  Lastly, PMpred
has currently only been tested with **Python 3.6+**.

The first two packages **numpy** and **scipy** are commonly used Python packages, and pre-installed on many computer systems. The last **joblib** package can be installed using **pip** [https://joblib.readthedocs.io/en/stable/](https://joblib.readthedocs.io/en/stable/), which is also pre-installed on many systems.

With these three packages in place, you should be all set to install and use PMpred.

### Installing PMpred

As with most Python packages, configurating LDpred is simple.  You can use **pip** to install it by typing

`pip install pmpred`

This should automatically take care of dependencies.  The examples below assume ldpred has been installed using pip.

Alternatively you can use **git** (which is installed on most systems) and clone this repository using the following git command:

`git clone https://github.com/WiuYuan/pmpred.git`

Then open the terminal of the repository folder and run command:

`pip install .`

Finally, you can also download the source files and place them somewhere.

With the Python source code in place and the three packages **numpy**, **scipy**, and **joblib** installed, then you should be ready to use PMpred.

## Using PMpred

A typical LDpred workflow consists of 3 steps:

### Step 1: Get data incude Precision Matrix, Snplists and GWAS Sumstats

The first step is to prepare the data we use in PMpred, contain {Precision Matrix, Snplists, GWAS Sumstats}

* Precision Matrix: could be download in [https://zenodo.org/records/8157131](https://zenodo.org/records/8157131)
* Snplists: could be download in [https://zenodo.org/records/8157131](https://zenodo.org/records/8157131)
* GWAS Sumstats: should be prepared using csv format with split `\t` and need include five head {rsid, REF, beta, beta_sd, N}. An example is showed below:
  
```{}
rsid    REF    ALT    beta    beta_sd    N ...
  *      *      *        *       *       *
  *      *      *        *       *       *
  *      *      *        *       *       *
...
```

Certainly, you can specify the headers of sumstats and split with parameters in pmpred like below:

```{}
--rsidname SNP
--REFname A1
--ALTname A2
--betaname BETA
--sename SE
--Nname n
--split ,
...
```

Then the sumstats could be like:

```{}
SNP,A1,A2,BEAT,SE,N,...
*,*,*,*,*,*,...
*,*,*,*,*,*,...
...
```

### Step 2: Choose the method using in PMpred

After getting the required data we could easily get the effect size using the quick start below:

```{bash}
pmpred --pm precision_matrix_folder --snp snplists_folder -s sumstats_file -o output_file
```

If you use precision matrix many times, you could first normalize it using command below:

```{bash}
pmpred --pm precision_matrix_folder -o new_precision_matrix_folder
```

then use pmpred without normalize Precision Matrix

```{bash}
pmpred --pm precision_matrix_folder --snp snplists_folder -s sumstats_file -o output_file --unnormal
```

Other parameters in PMpred could be found in

```{bash}
pmpred -h
```
