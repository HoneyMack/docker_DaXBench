#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import io
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")) as fh:
        return fh.read()


setup(
    name="daxbench_gymwrapper",
    version="0.0.0",
    packages=find_packages(where="daxbench_gymenv"),
    package_dir={"": "daxbench_gymenv"},
    # packages=find_packages(),
    # package_dir={"": "."},
    python_requires=">=3.8",
    include_package_data=True,
)


# setup(
#     name="daxbench_gymwrapper",
#     version="0.0.0",
#     license="BSD-2-Clause",
#     description="",
#     long_description="",
#     packages=find_packages(),
#     package_dir={"": "."},
#     py_modules=[splitext(basename(path))[0] for path in glob("daxbench/*.py")],
#     include_package_data=True,
#     zip_safe=False,
#     classifiers=[
#         # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
#         "Development Status :: 1 - Planning",
#         "Intended Audience :: Developers",
#         "License :: OSI Approved :: BSD License",
#         "Operating System :: Unix",
#         "Programming Language :: Python",
#         "Programming Language :: Python :: 3",
#         "Programming Language :: Python :: 3 :: Only",
#         "Programming Language :: Python :: 3.8",
#         "Programming Language :: Python :: 3.9",
#         "Programming Language :: Python :: Implementation :: CPython",
#         "Programming Language :: Python :: Implementation :: PyPy",
#         # uncomment if you test on these interpreters:
#         # 'Programming Language :: Python :: Implementation :: IronPython',
#         # 'Programming Language :: Python :: Implementation :: Jython',
#         # 'Programming Language :: Python :: Implementation :: Stackless',
#         "Topic :: Utilities",
#         "Private :: Do Not Upload",
#     ],
#     keywords=[
#         # eg: 'keyword1', 'keyword2', 'keyword3',
#     ],
#     python_requires=">=3.8",
#     install_requires=[],
#     extras_require={
#         "dev": ["pytest"]
#         # eg:
#         #   'rst': ['docutils>=0.11'],
#         #   ':python_version=="2.6"': ['argparse'],
#     },
#     entry_points={
#         "console_scripts": [
#             "daxbench = daxbench.cli:main",
#         ]
#     },
# )
