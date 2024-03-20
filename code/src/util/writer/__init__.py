from collections import OrderedDict
from .image import ImageWriter
import logging

log = logging.getLogger(__name__)


def build_image_writer(engine_config: OrderedDict):
    log.info("Building image writer")
    return ImageWriter(engine_config["image_writer"])
