# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="fast3r",
    version="1.0",
    description="Fast3R: Fast 3D Reconstruction",
    author="Jianing Yang",
    author_email="jianingy@umich.edu",
    url="https://fast3r-3d.github.io/",
    packages=find_packages(include=["fast3r"]),
)
