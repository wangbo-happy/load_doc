from setuptools import setup, find_packages
import os

# Read the contents of README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Read version from __init__.py
with open('src/__init__.py', encoding='utf-8') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip("'\"")
            break
    else:
        version = '0.1.0'

setup(
    name="data_analysis_toolkit",
    version=version,
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive toolkit for data processing and analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/data_analysis_toolkit",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "data_analysis=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.ini"],
    },
) 