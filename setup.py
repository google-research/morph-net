"""Setup for morph-net package."""

import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
    name="morph_net",
    version="0.2.1",
    author="Google LLC",
    author_email="morphnet@google.com",
    description="A library for learning deep network structure during training",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/morph-net",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
