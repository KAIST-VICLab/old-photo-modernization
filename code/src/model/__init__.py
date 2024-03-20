import importlib
from .base import BaseModel


def find_model_class(model_file, model_class):
    model_filename = "src.model.{}".format(model_file)
    modellib = importlib.import_module(model_filename)
    model = None
    for name, cls in modellib.__dict__.items():
        if name == model_class and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        raise NotImplementedError(
            "In %s.py, there should be a subclass of BaseModel with class name "
            "that matches %s in lowercase" % (model_filename, model_class)
        )

    return model


def build_model(
    model_config, phase, device, gpu_ids, verbose, is_distributed: bool = False
):
    model = find_model_class(model_config["file"], model_config["class"])
    instance = model(model_config, phase, device, gpu_ids, verbose, is_distributed)
    return instance
