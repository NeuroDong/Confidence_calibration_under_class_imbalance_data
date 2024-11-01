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
task_mode = "Post-hoc Calibration"
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "temperature_scale_with_ece"
_C.MODEL.NUM_CLASS = 100
_C.MODEL.DEVICE = "cuda:0"
_C.MODEL.JUST_EVAL = False
_C.MODEL.PRE_WEIGHT = False
_C.MODEL.PREWEIGHT = ""

_C.SOLVER = CN()
_C.SOLVER.START_ITER = 0
_C.SOLVER.OPTIMIZER = "SGD"
_C.SOLVER.LR_SCHEDULER_NAME = "Step_Decay"
_C.SOLVER.BASE_LR = 0.01    #Corresponds to the method in _C.MODEL.META_ARCHITECTURES.
_C.SOLVER.MOMENTUM = 0.9      
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.NESTEROV = False  
_C.SOLVER.GAMMA = 0.1 
_C.SOLVER.STEPS = (250,) 
_C.SOLVER.CLR_STEPS = 2000     #If the CLR learning rate schedule is used, this parameter needs to be set.
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.MAX_ITER = 64000
_C.SOLVER.MAX_EPOCH = 500
_C.SOLVER.BATCHSIZE = 1000
_C.SOLVER.NUM_WORKS = 20
_C.SOLVER.BETA = (0.9, 0.999)   #default: (0.9, 0.999)

_C.DATASET = CN()
_C.DATASET.SCENARIOS = "General data"
# General data
'''
resnet110_c10_logits.p
resnet110_SD_c10_logits.p
resnet_wide32_c10_logits.p
densenet40_c10_logits.p
resnet110_c100_logits.p
resnet110_SD_c100_logits.p
resnet_wide32_c100_logits.p
densenet40_c100_logits.p
resnet152_imgnet_logits.p
densenet161_imgnet_logits.p
'''

# Label shift data
pass

# Coviarate shift data
pass

# OOD data
pass

# set path
_C.DATASET.DATA_PATH = r'datasets/public_logits_datasets/densenet161_imgnet_logits.p'
_C.EVALUATE = CN()
_C.EVALUATE.METHOD_list = ["ECE_with_equal_mass", "Debaised_ECE", "ECE_sweep_em", "KS_error", "SmoothingECE","TCE_BPM"]  #"TCE_BPM"
_C.MODEL.OUTPUT_DIR = "output/"+task_mode+"/"+_C.DATASET.SCENARIOS+"/"+_C.DATASET.DATA_PATH.split("/")[-1][:-2]+"/"+ _C.MODEL.META_ARCHITECTURE + "/"