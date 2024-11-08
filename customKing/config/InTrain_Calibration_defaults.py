from .config import CfgNode as CN

# ----------------------------------------------------------------------------- #
# Convention about Training / Test specific parameters
# ----------------------------------------------------------------------------- #
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# ----------------------------------------------------------------------------- #
# Config definition
# ----------------------------------------------------------------------------- #
_C = CN()
_C.SEED = 20

# ---------------------------------------------------------------------------- #
# Calibration model config
# ---------------------------------------------------------------------------- #
task_mode = "In-Train Calibration"
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "Wide_resnet34_2_DualFocalLoss"
_C.MODEL.NUM_ClASSES = 2
_C.MODEL.DEVICE = "cuda:0"
_C.MODEL.JUST_EVAL = False
_C.MODEL.PRE_WEIGHT = False
_C.MODEL.PREWEIGHT = ""

_C.SOLVER = CN()
_C.SOLVER.START_ITER = 0
_C.SOLVER.OPTIMIZER = "SGD"
_C.SOLVER.LR_SCHEDULER_NAME = "Step_Decay"
_C.SOLVER.BASE_LR = 0.1    #Corresponds to the method in _C.MODEL.META_ARCHITECTURES.
_C.SOLVER.MOMENTUM = 0.9      
_C.SOLVER.WEIGHT_DECAY = 5./10000
_C.SOLVER.NESTEROV = False  
_C.SOLVER.GAMMA = 0.1 
_C.SOLVER.STEPS = (32000,48000) 
_C.SOLVER.CLR_STEPS = 2000     #If the CLR learning rate schedule is used, this parameter needs to be set.
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.MAX_ITER = 64000
_C.SOLVER.MAX_EPOCH = 500
_C.SOLVER.BATCH_SIZE = 128
_C.SOLVER.NUM_WORKS = 20
_C.SOLVER.BETA = (0.9, 0.999)   #default: (0.9, 0.999)
_C.SOLVER.SHUFFLE = True 
_C.SOLVER.NUM_WORKERS = 8    #the num workers of the Dataloader
_C.SOLVER.IS_PARALLEL = False   #Whether to use multiple GPUs for training

_C.DATASETS = CN()
_C.DATASETS.TRAIN = "Survival_train"    #train dataset
_C.DATASETS.VALID = "Survival_valid"    #valid dataset
_C.DATASETS.TEST = "Survival_test"      #test dataset

_C.EVALUATE = CN()
_C.EVALUATE.METHOD_list = ["ECE_with_equal_mass", "Debaised_ECE", "ECE_sweep_em", "KS_error", "SmoothingECE"]  #"TCE_BPM"
_C.MODEL.OUTPUT_DIR = "output/"+task_mode+"/"+_C.DATASETS.TRAIN.split("_")[0] + "/" + _C.MODEL.META_ARCHITECTURE + "/"