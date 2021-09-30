"""Utils to generate register and loader for Loss, Model and BatchLoader"""
from functools import partial


def _factory_add(name_mapping, force_base_class=object):
    """A function factory to generate functions access to different mapping

    Args:
        name_mapping (dict): the name => class mapping
        force_base_class (class, optional): type-check the candidate to be inherited
            from this base class. Will always pass the check if use default `object`
            value.

    Returns:
        add_func (function): the add_xxx function within the closure of name_mapping
    """
    def add_func(name, some_cls=None):
        """Add a customized class to the namespace

        It may be used as a function or a decorator of class definition.
        `some_cls` maybe Loss, Model, Collector and etc.
        """
        # TO support @add_func('XXX') style registration
        if some_cls is None:
            return partial(add_func, name)

        if name in name_mapping:
            raise ValueError(f'(name={name}) already added')
        # Any class will be sub-class of `object` (by default)
        if not issubclass(some_cls, force_base_class):
            raise ValueError(f'(class={some_cls}) must be sub-class of {force_base_class}')
        name_mapping[name] = some_cls
        # FIXME: cannot log before asr.launch is initialized
        # logging.info(f'{name}={some_cls} registered')
        return some_cls

    return add_func


def _factory_build(name_mapping):
    """A function factory to generate functions access to different mapping

    Args:
        name_mapping (dict): the name => class mapping

    Returns:
        build_func (function): the build_xxx function within the closure of name_mapping
    """
    def build_func(hparams):
        """Build loss/model/batch_loader given dynamic hyper parameters

        .. warning:: NO deepcopy is allowed for the hparams

        Args:
            hparams (dict): hyper parameters

        Returns:
            object (cls in name_mapping)
        """
        if not isinstance(hparams, dict):
            raise ValueError(f'Wrong type of parameters passed in to build, expected dict, got {type(hparams)}')
        # hparams should not be modified, here we make a shallow copy
        # in build_scheduler, there's an optimizer object passed in, we cannot deepcopy it otherwise
        # lr_scheduler will not work
        hparams_copy = hparams.copy()
        dynamic_obj_type = hparams_copy.pop('name')
        try:
            dynamic_obj_cls = name_mapping[dynamic_obj_type]
        except KeyError:
            raise KeyError(f'Name `{dynamic_obj_type}` not found in mapping {name_mapping}.'
                           'You may need to check if the extended modules are imported before launching')
        dynamic_obj = dynamic_obj_cls(**hparams_copy)
        return dynamic_obj

    return build_func
