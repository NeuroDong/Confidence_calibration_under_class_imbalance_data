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
task_mode = "Main"

_C.CALIBRATION = CN()
_C.CALIBRATION.MODE = "In-train"   # Data_level, In-train, or Post-hoc
_C.CALIBRATION.NEED_CLASSIFIER_TRAIN = True
_C.CALIBRATION.NEED_CALIBRATION_TRAIN = False
if _C.CALIBRATION.NEED_CALIBRATION_TRAIN:
    _C.CALIBRATION.ITERATIVE_TRAINING = False
 
_C.MODEL = CN()
_C.MODEL.META_ARCHITECTURE = "ResNet110_TLCLoss"
_C.MODEL.NUM_CLASS = 10
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
_C.SOLVER.WEIGHT_DECAY = 2e-4
_C.SOLVER.NESTEROV = False  
_C.SOLVER.GAMMA = 0.1 
_C.SOLVER.STEPS = (6000,13000) 
_C.SOLVER.CLR_STEPS = 2000     #If the CLR learning rate schedule is used, this parameter needs to be set.
_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.MAX_ITER = 64000
_C.SOLVER.MAX_EPOCH = 150
_C.SOLVER.BATCH_SIZE = 128
_C.SOLVER.NUM_WORKS = 8
_C.SOLVER.BETA = (0.9, 0.999)   #default: (0.9, 0.999)
_C.SOLVER.SHUFFLE = True 
_C.SOLVER.PIN_MEMORY = False

_C.EVALUATE = CN()
_C.EVALUATE.METHOD_list = ["ECE_with_equal_mass","ICE_smooth","ICE_soft","ICE_strong","CECE","MSECE","WSECE","RBECE"]  #"TCE_BPM"

_C.DATASET = CN()
_C.DATASET.IS_LOGITS_DATA = False
if _C.DATASET.IS_LOGITS_DATA:
    # Public logits data
    '''
    public_logits_datasets/resnet110_c10_logits.p
    public_logits_datasets/resnet110_SD_c10_logits.p
    public_logits_datasets/resnet_wide32_c10_logits.p
    public_logits_datasets/densenet40_c10_logits.p
    public_logits_datasets/resnet110_c100_logits.p
    public_logits_datasets/resnet110_SD_c100_logits.p
    public_logits_datasets/resnet_wide32_c100_logits.p
    public_logits_datasets/densenet40_c100_logits.p
    public_logits_datasets/resnet152_imgnet_logits.p
    public_logits_datasets/densenet161_imgnet_logits.p
    '''

    # Generate logits data
    '''
    Generate_logits_datasets/Credit_MLPClassifier.p
    Generate_logits_datasets/Creditcard_MLPClassifier.p
    '''
    _C.DATASET.SCENARIOS = "Generate_logits"
    # set path
    _C.DATASET.DATA_PATH = r'Generate_logits_datasets/Creditcard_MLPClassifier.p'
    _C.MODEL.OUTPUT_DIR = "output/"+_C.CALIBRATION.MODE+"/"+_C.DATASET.SCENARIOS+"/"+_C.DATASET.DATA_PATH.split("/")[-1][:-2]+"/"+ _C.MODEL.META_ARCHITECTURE + "/"
else:
    _C.DATASET.SAMPLER = None  # undersampler
    #Select the dataset name from customKing/data/datasets/builtin.py
    _C.DATASET.TRAIN = "Cifar10_train_LT"    #train dataset
    _C.DATASET.VALID = "Cifar10_valid_LT"    #valid dataset
    _C.DATASET.TEST = "Cifar10_test"      #test dataset
    _C.DATASET.INPUTE_SIZE = (32,32)
    _C.MODEL.OUTPUT_DIR = "output/"+_C.CALIBRATION.MODE+"/"+_C.DATASET.TRAIN.split("_")[0] + "/" + _C.MODEL.META_ARCHITECTURE + "/"

