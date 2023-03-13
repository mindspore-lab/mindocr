#!/usr/bin/env python

from setuptools import find_packages, setup

__version__ = '0.1'

setup(
    name="mindocr",
    author="MindSpore Ecosystem",
    author_email="mindspore-ecosystem@example.com",
    url="https://github.com/mindspore-lab/mindocr",
    project_urls={
        "Sources": "https://github.com/mindspore-lab/mindocr",
        "Issue Tracker": "https://github.com/mindspore-lab/mindocr/issues",
    },
    description="A toolbox of vision models and algorithms based on MindSpore.",
    license="Apache Software License 2.0",
    include_package_data=True,
    package_dir={'mindocr.mindocr': 'mindocr', 'mindocr.tools': 'tools', 'mindocr.deploy': 'deploy'},
    entry_points={"console_scripts": ["mindocr=mindocr.deploy.mx_infer.infer_pipeline:main"]},
    install_requires=[
        "numpy >= 1.17.0",
    ],
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
    ],
    test_suite="tests",
    tests_require=[
        "pytest",
    ],
    version=__version__,
    zip_safe=False,
)