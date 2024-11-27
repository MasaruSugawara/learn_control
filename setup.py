from setuptools import setup, find_packages

setup(
    name='ctrl_sim',
    version="0.1.0",
    description="A simulator system of controlling dynamical systems just for learning",
    author='MasaruSugawara',
    packages=find_packages(),
    license='Apache-2.0',
    url='https://github.com/MasaruSugawara/learn_control.git',
    install_requires=[
      'numpy', 'casadi', 'matplotlib'
      ],
    python_requires='>=3.6'
)
