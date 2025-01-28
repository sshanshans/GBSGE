from setuptools import setup, find_packages

setup(
    name='GBSGE',
    version='0.1.0',
    description='Use Gaussian Boson Sampling for Approximate Gaussian Expectation Problems',
    url='https://github.com/sshanshans/GBEGE.git',
    author='Shan Shan',
    author_email='shan-qm@imada.sdu.dk',
    license='GPL-3.0',
    packages=find_packages(),
    install_requires=[
        # Project dependencies.
        'thewalrus', 
        'joblib',
        'tabulate',
        'matplotlib',
        'scipy',
    ],
    zip_safe=False
)

