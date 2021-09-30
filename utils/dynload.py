"""A centralized collection for dynload functions {add/build}_{loss/model/collector} for backward compatibility"""
# pylint: disable=unused-import
import warnings

DYNLOAD_DEPRECATION_WARNING = (
    f'The {__name__} module is under deprecation, the add/build function is moved to the corresponding '
    'module. try to use `from model import add_model` instead.'
)
warnings.warn(DYNLOAD_DEPRECATION_WARNING, DeprecationWarning)

from ..data.collector import add_collector, build_collector
from ..loss import add_loss, build_loss
from ..model import add_model, build_model
from ..trainer.lr_scheduler import add_scheduler, build_scheduler
from .dynload_factory import _factory_add, _factory_build
