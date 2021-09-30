"""Maintain the metadata of this package and import some important sub-module"""

# Pre-import some important modules
from . import (
    data, loss, model, utils, trainer
)


# Version priority: git > package > release
try:
    from setuptools_scm import get_version
    __version__ = get_version(root='..', relative_to=__file__)
except LookupError:
    from pkg_resources import get_distribution, DistributionNotFound
    try:
        __version__ = get_distribution('pytorch-asr').version
    except DistributionNotFound:
        # package is not installed
        __version__ = 'unknown'
