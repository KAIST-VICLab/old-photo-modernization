from torch import nn
import logging

log = logging.getLogger(__name__)


def get_pad_layer(pad_type):
    PadLayer = None
    if pad_type in ["refl", "reflect"]:
        PadLayer = nn.ReflectionPad2d
    elif pad_type in ["repl", "replicate"]:
        PadLayer = nn.ReplicationPad2d
    elif pad_type == "zero":
        PadLayer = nn.ZeroPad2d
    else:
        log.info("Pad type [%s] not recognized" % pad_type)
    return PadLayer
