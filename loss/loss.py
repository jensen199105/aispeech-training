import torch.nn as nn
from ..utils.dynload_factory import _factory_add, _factory_build


class Loss(nn.Module):
    """Base Loss class

    Loss class should compute the overall loss value and compose a log_stat dict
    with all the loss information
    """
    def forward(self, model_output, batch):
        """Loss forward function

        .. warning::
            The returned loss_stat must give the correct ``loss`` and ``total_frames``
            value to make distributed training and learning rate scheduling work properly.

        Args:
            model_output: the model output, usually a :class:`~asr.data.field.Field`
                          but can be arbitrary type
            batch (:class:`~asr.data.field.Batch`): data batch

        Returns:
            loss (torch.Tensor): a scalar loss tensor, must be available to call
                                 `.backward()`
            loss_stat (dict): a log statistics information dict, it must have ``loss``
                              and ``total_frames`` keys at least.
        """
        raise NotImplementedError

    def log_line(self, reduced_stat):
        """Print customized error message

        This method will be called after every epoch to get the useful loss
        information given by the inherited loss type.
        The trainer will accumulate (+ operator) the loss_statistics returned by
        the loss.

        Args:
            reduced_stat (dict): the loss statistics with its' values reduced

        Returns:
            log (str): A one-liner log information

        """
        raise NotImplementedError('The Loss framework must implement `log_line` method')


_loss_mapping = {}

add_loss = _factory_add(_loss_mapping, force_base_class=Loss)
add_loss.__doc__ = """Add customized loss to dynamic load

See :ref:`notes/dynload:Dynload string format` for detailed format.

Built-in loss: {'ce', 'ctc', 'mmi'}

Usages:
    1. use as a decorator (recommended), remember to ``import``
       before launching.

       .. code-block::
           @add_loss('dummy_loss')
           class DummyLoss(Loss):
               pass

    2. use as a function

       .. code-block::
           DummyLoss = Loss
           add_loss('dummy_loss', DummyLoss)

Args:
    name (str): the command line name
    some_cls (:class:`~asr.loss.Loss`): customized loss
"""
build_loss = _factory_build(_loss_mapping)
build_loss.__doc__ = """Build the loss given dynload dict

See :ref:`notes/dynload:Dynload string format` for detailed format.

Args:
    hparams (dict): the dict built from dynload string.

Returns:
    :class:`~asr.loss.Loss`: loss of class ``hparams['name']``.
"""

