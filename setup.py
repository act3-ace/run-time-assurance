#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Setup script for run-time-assurance"""
from pathlib import Path

from setuptools import setup, find_packages


def parse_requirements(filename: str):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

reqs = parse_requirements("requirements.txt")


if __name__ == '__main__':
    tests_require = [
        'flake8',
        'mypy',
        'mypy-extensions',
        'mypy-protobuf',
        'pylint',
        'yapf',
        'isort',
        'rope',
        'pre-commit',
        'pre-commit-hooks',
        'detect-secrets',
        'blacken-docs',
        'bashate',
        'fish',
        'watchdog',
        'speedscope',
        'pandas-profiling',
        'factory',
    ]

    docs_require = [
        'mkdocs',
        'mkdocs-macros-plugin',
        'mkdocs-mermaid-plugin',
        'inari[mkdocs]',
        'pymdown-extensions',
    ]

    setup(
        name="run-time-assurance",
        author="ACT3 Safe Autonomy Team",
        description="ACT3 Safe Autonomy Run Time Assurance Package",
        version="0.1.1",

        long_description=Path("README.md").read_text(),
        long_description_content_type="text/markdown",

        url="https://github.com/act3-ace/run-time-assurance",

        license="",

        packages=find_packages(),

        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],

        install_requires=reqs,

        extras_require={
            "testing":  tests_require,
            "docs":  docs_require,
        },
        python_requires='>=3.7',
    )
