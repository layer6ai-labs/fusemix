"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from setuptools import setup, find_namespace_packages
import platform

DEPENDENCY_LINKS = []
if platform.system() == "Windows":
    DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")


def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n")]


setup(
    name="fusemix",
    version="1.0",
    author="Layer 6 AI",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_namespace_packages(include="lavis.*"),
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.7.0",
    include_package_data=True,
    dependency_links=DEPENDENCY_LINKS,
    zip_safe=False,
)
