import random
import numpy 
import os


def frozen(seed):
    numpy.random.seed(seed)
    random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

    try:
        import tensorflow
        tensorflow.random.set_seed(seed)
        os.environ["TF_DETERMINISTIC_OPS"] = str(seed)
        os.environ["PYTHONASHSEED"] = str(seed)
    except ImportError:
        pass



