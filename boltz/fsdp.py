from torch.distributed import DeviceMesh
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)


def fsdp_auto_wrap_policy(model, transformer_layer_names):
    import functools

    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn
    )
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=set(transformer_layer_names)
    )

    auto_wrap_policy = functools.partial(
        _or_policy, policies=[lambda_policy, transformer_wrap_policy]
    )
    return auto_wrap_policy
