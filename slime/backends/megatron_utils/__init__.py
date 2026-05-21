import logging

import torch

from slime.utils.common import is_npu
if is_npu():
    import mindspeed.megatron_adaptor

try:
    import deep_ep
    from torch_memory_saver import torch_memory_saver

    old_init = deep_ep.Buffer.__init__

    def new_init(self, *args, **kwargs):
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
        old_init(self, *args, **kwargs)
        torch.cuda.synchronize()
        if torch_memory_saver._impl is not None:
            torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

    deep_ep.Buffer.__init__ = new_init
except ImportError:
    logging.warning("deep_ep is not installed, some functionalities may be limited.")

try:
    from mbridge.models.qwen3_vl.model import Qwen3VLModel
    _original_forward2 = Qwen3VLModel.forward

    def _patched_forward2(self, *args, loss_mask=None, **kwargs):
        return _original_forward2(self, *args, **kwargs)
    Qwen3VLModel.forward = _patched_forward2
except ImportError:
    pass

logging.getLogger("megatron").setLevel(logging.WARNING)
