import os
import pathlib
import pkg_resources
from setuptools import setup, find_packages


PKG_NAME = "diversevla"
VERSION = "0.1"
EXTRAS = {}

def _read_file(fname):
    with pathlib.Path(fname).open(encoding="utf-8") as fp:
        return fp.read()


def _read_install_requires():
    with pathlib.Path("requirements.txt").open() as fp:
        return [
            str(requirement) for requirement in pkg_resources.parse_requirements(fp)
        ]


def _fill_extras(extras):
    if extras:
        extras["all"] = list(set([item for group in extras.values() for item in group]))
    return extras

setup(
    name=PKG_NAME,
    version=VERSION,
    author=f"Ran Zhu, Muyao Li",
    url="https://github.com/MuyaoLi-jimo/DiverseVLA.git",
    description="research project",
    long_description=_read_file("README.md"),
    long_description_content_type="text/markdown",
    keywords=[
        "Open-ended Learning",
        "Embodied Agents",
        "Large Language Models",
    ],
    license="MIT License",
    packages=find_packages(include=f"{PKG_NAME}.*"),
    include_package_data=True,
    zip_safe=False,
    install_requires=_read_install_requires(),
    extras_require=_fill_extras(EXTRAS),
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3.9",
    ],
)

