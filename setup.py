import os
from setuptools import setup, find_packages

# Safely read requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

# Safely read README.md for the long description (if it exists)
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="mann_engram_en",
    version="0.1.0",
    author="MANN-Engram Contributors",
    author_email="your.email@example.com", # TODO: 替换为您的邮箱
    description="An Edge-Cloud Multimodal Semantic Router for Medical AI & Noise Filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/MANN-Engram", # TODO: 替换为您的 GitHub 仓库地址
    packages=find_packages(include=["mann_engram_en", "mann_engram_en.*"]),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    include_package_data=True,
)