from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="proofflow",
    version="1.0.0",
    author="ProofFlow Team",
    author_email="",
    description="Automated mathematical proof formalization using Large Language Models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Reports": "",
        "Source": "",
        "Documentation": "",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "proofflow=proofflow.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "proofflow": [
            "prompts/*.md",
            "data/*.json",
            "data/*.xlsx",
        ],
    },
    keywords=[
        "mathematics",
        "formal-verification",
        "lean4",
        "proof-assistant",
        "large-language-models",
        "artificial-intelligence",
        "automated-theorem-proving",
        "mathematical-proofs",
    ],
    zip_safe=False,
)