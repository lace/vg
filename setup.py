import importlib
from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    install_requires = f.read()

setup(
    name="vector_shortcuts",
    version=importlib.import_module("vx").__version__,
    description="Vector shortcuts for NumPy",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Metabolize",
    author_email="github@paulmelnikow.com",
    url="https://github.com/metabolize/vx",
    project_urls={
        "Issue Tracker": "https://github.com/metabolize/vx/issues",
        "Documentation": "https://vx.readthedocs.io/en/stable/",
    },
    packages=find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Manufacturing",
        "Topic :: Artistic Software",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
)
