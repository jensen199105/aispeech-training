from argparse import Namespace

from omegaconf import OmegaConf
from .asr_config import ASRConfig


# FIXME: temporary hacking
def convert_omega_to_namespace(config: ASRConfig) -> Namespace:
    """Flatten the extra hierachy into conventional arg parser style"""
    namespace_args = Namespace()
    extra_hierachy = ['data', 'dist', 'trainer', 'optim']

    def flatten_param(config):
        for k, v in config.items():
            # DO NOT assign the extra hierachy to NameSpace for simplicity
            if k in extra_hierachy:
                continue
            try:
                v = OmegaConf.to_container(v, resolve=True)
            except AssertionError:
                pass
            setattr(namespace_args, k, v)

    flatten_param(config)
    for key in extra_hierachy:
        flatten_param(config[key])

    return namespace_args


def parse_omega_args(cli_args: Namespace) -> Namespace:
    schema = OmegaConf.structured(ASRConfig)
    conf_list = [schema]
    conf_list += [OmegaConf.load(conf_file) for conf_file in cli_args.conf]
    conf_list.append(OmegaConf.from_dotlist(cli_args.overrides))
    conf = OmegaConf.merge(*conf_list)

    raw_args = convert_omega_to_namespace(conf)
    return raw_args
