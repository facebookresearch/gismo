# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
    description="Inverse Cooking 2.0: Graph-based Ingredient Substitution Module",
    version="0.1",
    author="Facebook AI Research",
    author_email="",
    license="CC BY-NC 4.0",
    url="https://github.com/facebookresearch/gismo",
    packages=["inv_cooking"],
    install_requires=get_requirements(),
    python_requires=">=3.7",
    extras_require={"dev": get_requirements_dev()},
)
