from setuptools import setup, find_packages

setup(
    name="ftse-market-analysis",
    version="0.1.0",
    author="Rajan",
    description="Unsupervised ML for FTSE Market Analysis",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "streamlit",
        "scipy",
        "joblib",
        "pyyaml"
    ],
    python_requires=">=3.8"
)