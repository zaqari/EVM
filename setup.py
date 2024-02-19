from setuptools import setup, find_packages

import os
lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = f"{lib_folder}/requirements.txt"
install_requires = [] # Here we'll add: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requires = f.read().splitlines()

setup(
    name="EVM",
    author='Zachary Rosen',
    version="1.0",
    install_requires=install_requires,
    long_description="""The following is a full implementation of the convergence-entropy measurement framework (here referred to as Entropy-conVergence Metric, or EVM) as described in Rosen and Dale 2023. We've taken strides to make this package as easy to implement as possible.\n Additional details can be found at https://github.com/zaqari/EVM""",
    long_description_content_type='text/x-rst'
)
