from imp import IMP_HOOK
from .SETR import Setr, Setr_DSFormer

def get_model(modelname="Unet", img_size=256, img_channel=1, classes=9, assist_slice_number=4):

    if modelname == "SETR":
        model = Setr(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    elif modelname == "SETR_DSFormer":
        model = Setr_DSFormer(n_channels=img_channel, n_classes=classes, imgsize=img_size)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model