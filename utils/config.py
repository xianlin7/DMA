# This file is used to configure the training parameters for each task

class Config_Synapse:
    data_path = "../../dataset/Synapse/"
    save_path = "./checkpoints/Synapse/"
    result_path = "./result/Synapse/"
    tensorboard_path = "./tensorboard/Synapse/"
    load_path = save_path + "/xxx.pth"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 4)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 14                 # the number of classes (background + foreground)
    img_size = 256               # the input size of model
    train_split = "train"        # the file name of training set
    val_split = "test"           # the file name of testing set
    test_split = "test"
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000               # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "patient"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "DMA"

class Config_INSTANCE:
    data_path = "../../../dataset/INSTANCE/"
    save_path = "./checkpoints/INSTANCE/"
    result_path = "./result/INSTANCE/"
    tensorboard_path = "./tensorboard/INSTANCE/"
    load_path = "./xxxx"

    workers = 1                  # number of data loading workers (default: 8)
    epochs = 400                 # number of total epochs to run (default: 400)
    batch_size = 8               # batch size (default: 8)
    learning_rate = 1e-4         # initial learning rate (default: 0.001)
    momentum = 0.9               # momentum
    classes = 2                  # the number of classes
    img_size = 256               # the input size of model
    train_split = "train"        # the file name of training set
    val_split = "val"
    test_split = "test"          # the file name of testing set
    crop = None                  # the cropped image size
    eval_freq = 1                # the frequency of evaluate the model
    save_freq = 2000             # the frequency of saving the model
    device = "cuda"              # training device, cpu or cuda
    cuda = "on"                  # switch on/off cuda option (default: off)
    gray = "yes"                 # the type of input image
    img_channel = 1              # the channel of input image
    eval_mode = "patient"        # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "DMA"

class Config_ACDC:
    data_path = "../../../dataset/cardiac/"
    save_path = "./checkpoints/ACDC/"
    result_path = "./result/ACDC/"
    tensorboard_path = "./tensorboard/ACDC/"
    load_path = "./xxxx"

    workers = 1                   # number of data loading workers (default: 8)
    epochs = 400                  # number of total epochs to run (default: 400)
    batch_size = 8                # batch size (default: 4)
    learning_rate = 1e-4          # initial learning rate (default: 0.001)
    momentum = 0.9                # momentum
    classes = 4                   # the number of classes
    img_size = 256                # the input size of model
    train_split = "trainofficial" # the file name of training set
    val_split = "valofficial"
    test_split = "testofficial"   # the file name of testing set
    crop = None                   # the cropped image size
    eval_freq = 1                 # the frequency of evaluate the model
    save_freq = 2000              # the frequency of saving the model
    device = "cuda"               # training device, cpu or cuda
    cuda = "on"                   # switch on/off cuda option (default: off)
    gray = "yes"                  # the type of input image
    img_channel = 1               # the channel of input image
    eval_mode = "patient"         # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "DMA"


# ==================================================================================================
def get_config(task="Synapse"):
    if task == "Synapse":
        return Config_Synapse()
    elif task == "INSTANCE":
        return Config_INSTANCE()
    elif task == "ACDC":
        return Config_ACDC()
    else:
        assert("We do not have the related dataset, please choose another task.")