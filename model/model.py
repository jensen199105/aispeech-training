"""Base model with API definition"""
import warnings

import torch.nn as nn

from ..utils.dynload_factory import _factory_add, _factory_build


class Model(nn.Module):
    """Base model for this framework, with extra APIs

    Model is a sub-class of ``nn.Module``, it adds more specific APIs for generic
    model integration in pytorch-asr framework.

    **Model** in this framework is responsible for:
        - (optionally) init the *torch.nn.Module* if necessary
        - unpack the data batch and do forward propagation
        - (optionally if inner skip) skip the label and add *skipped_label* field
            into data batch **inplace**
        - (optionally) post-process the gradient (e.g. clip_value in LSTM)
    """

    def extra_repr(self):
        total_param = sum([param.numel() for param in self.parameters()])
        trainable_param = sum([param.numel() for param in self.parameters() if param.requires_grad])
        extra_repr = f'total params: {total_param:,d}'
        extra_repr += f'\ntrainable params: {trainable_param:,d}'
        return extra_repr

    def forward(self, data_batch):
        """The forward computation of Model

        All models must implement this method, the same as ``nn.Module`` .

        Args:
            data_batch (:class:`~asr.data.Batch`): the data batch dict

        Returns:
            :class:`~asr.data.Field`: the output field, it can also be a tuple of Field.
        """
        raise NotImplementedError('Please implement your own forward')

    def grad_post_processing(self):
        """Post-process the gradient

        The method will be called after back-propagation, you must modify the
        gradient **inplace**.
        """
        warnings.warn('Gradient is not processed')

    def init_parameters(self):
        """Initialize the model parameters

        The method will be called right after model initialization in trainer,
        you can also call this method in *__init__* for any reasons.
        """
        warnings.warn('Model parameters are not manually initialized')

    def decode(self, data_batch):
        """An individual forward method ONLY for decoding

        In simple cases you don't have to re-implement this method. But if the training stage
        and decoding stage differs, please implement this in sub-classes.

        The method will be called in at decode.py.

        Args:
            - data_batch (Batch):
        """
        warnings.warn('`decode` method not found, falling back to `forward`')
        return self.forward(data_batch)


_model_mapping = {}

add_model = _factory_add(_model_mapping, force_base_class=Model)
add_model.__doc__ = """Add customized model to dynamic load

Usages:
    1. use as a decorator (recommended), remember to ``import``
       before launching.

       .. code-block::
           @add_model('dummy_model')
           class DummyModel(Model):
               pass

    2. use as a function

       .. code-block::
           DummyModel = Model
           add_model('dummy_model', DummyModel)

Args:
    name (str): the command line name
    some_cls (:class:`~asr.model.Model`): customized model
"""
build_model = _factory_build(_model_mapping)
build_model.__doc__ = """Build the model given dynload string

See :ref:`notes/dynload:Dynload string format` for detailed format.

Built-in model: {'FSMN', 'LSTM', 'CLD', 'Transformer'}

Args:
    hparams (dict): the dict built from dynload string.

Returns:
    :class:`~asr.model.Model`: built model of class ``hparams['name']``.
"""

