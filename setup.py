from setuptools import setup, find_packages

setup(
    name="adaptive_bayesian_driver",
    version="1.0.0",
    description="An adaptive learning agent based on a recursive Bayesian framework.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "Pillow",
        "PyYAML",
        "matplotlib"
    ],
    entry_points={
        "console_scripts": [
            "adaptive-bayesian-driver=adaptive_bayesian_driver.main:main"
        ]
    }
)
