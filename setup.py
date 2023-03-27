import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mmf-lp-modes",
    version="0.1",
    author="Jysru",
    author_email="jysru@pm.me",
    description="MMF_LP_modes_simulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jysru/MMF_LP_modes",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)