import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PMpred",
    version="1.0.4",
    url="https://github.com/WiuYuan/pmpred",
    author="Wen Yuan",
    author_email="wiuwenyuan@gmail.com",
    description="A Python package that adjusts GWAS summary statistics for the effects of Sparse Precision Matrix (PM)",
    keywords="Polygenic Risk Scores, GWAS, Linkage Disequilibrium, Risk Prediction, Precision Matrix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=["numpy", "scipy", "joblib"],
    entry_points={
        "console_scripts": [
            "pmpred=pmpred.main:main",
        ],
    },
)
