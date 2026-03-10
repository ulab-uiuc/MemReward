'''Setup script for efficient_reward_graph package.'''

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README_VERL_Integration.md").read_text()

setup(
    name="efficient_reward_graph",
    version="0.2.0",  # v0.2.0: VERL integration
    author="Efficient Reward Graph Team",
    description="Efficient reward computation for RLHF training using graph neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "verl": [
            "verl==0.6.1",
            "ray>=2.0.0",
            "flash-attn==2.7.0.post2",
            "ninja==1.13.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
