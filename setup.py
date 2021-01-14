# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pathlib

import pkg_resources
from setuptools import setup


def get_requirements():
    with pathlib.Path("requirements.txt").open() as requirements_txt:
        return [
            str(requirement)
            for requirement in pkg_resources.parse_requirements(requirements_txt)
        ]


def get_requirements_dev():
    with pathlib.Path("requirements_dev.txt").open() as requirements_txt:
        return [
            str(requirement)
            for requirement in pkg_resources.parse_requirements(requirements_txt)
        ]


setup(
    name="Inverse Cooking 2.0",
    description="Inverse Cooking 2.0",
    version="0.1",
    author="Facebook AI Research",
    author_email="",
    license="MIT",
    url="https://github.com/fairinternal/inversecooking2.0",
    packages=["inv_cooking"],
    install_requires=get_requirements(),
    python_requires=">=3.7",
    extras_require={"dev": get_requirements_dev()},
)
