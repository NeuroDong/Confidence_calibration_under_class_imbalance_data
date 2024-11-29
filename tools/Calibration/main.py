import torch
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
sys.path.insert(0,parentdir) 
from customKing.utils.env import Set_seed
from customKing.config.config import get_cfg
from customKing.modeling.meta_arch.build import build_model,META_ARCH_REGISTRY
from customKing.data.datasets.Public_logits_data.unpickle_probs import unpickle_probs
from customKing.data.datasets.Generate_logits_data.unpickle_generate_logits import unpickle_g_logits
import logging
from customKing.data import get_dataset_dicts,build_loader,build_sampling_loader
from compute_metrics import compute_metrics
from Trainer import doTrain


def main(Specific_cfg_path):
    
    #-----------Load and save config----------------#
    Set_seed(seed=20)
    task_mode = "Main"
    cfg = get_cfg(task_mode)
    if not os.path.exists(cfg.MODEL.OUTPUT_DIR):
        os.makedirs(cfg.MODEL.OUTPUT_DIR)
    if Specific_cfg_path != None:
        cfg.merge_from_file(Specific_cfg_path)
    #save cfg
    cfg_save_path = os.path.join(os.path.join(r"Cfg",cfg.CALIBRATION.MODE),cfg.MODEL.META_ARCHITECTURE+'.yaml')
    if os.path.exists(cfg_save_path):
        os.remove(cfg_save_path)
    with open(cfg_save_path, 'a+') as f:
        print(cfg, file=f)

    #-----------Load Data------------------#
    if cfg.DATASET.IS_LOGITS_DATA:
        _root = os.path.expanduser(os.getenv("CUSTOM_KING_DATASETS", "datasets"))
        data_path = os.path.join(_root,cfg.DATASET.DATA_PATH)
        if "public_logits_datasets" in data_path:
            valid_data, test_data = unpickle_probs(data_path, False)
            train_data = None
        else:
            train_data, valid_data, test_data = unpickle_g_logits(data_path,False)
        cls_num_list_train = None
        cls_num_list_test = None
    else:
        train_data = build_sampling_loader(cfg, get_dataset_dicts(cfg.DATASET.TRAIN))
        cls_num_list_train = train_data.dataset.get_cls_num_list()
        valid_data = build_loader(cfg, get_dataset_dicts(cfg.DATASET.VALID))
        test_data = build_loader(cfg, get_dataset_dicts(cfg.DATASET.TEST))
        cls_num_list_test = test_data.dataset.get_cls_num_list()

    #-----------Load model-----------------#
    if cfg.MODEL.PRE_WEIGHT:
        model = torch.load(cfg.MODEL.PREWEIGHT)
    else:
        model = build_model(cfg,cls_num_list_train,cls_num_list_test)
    logging.info(f"Calibration method:{cfg.MODEL.META_ARCHITECTURE}")
    model.to(cfg.MODEL.DEVICE)

    #-----------Compute Metrics Before Calibration-----------#
    if cfg.CALIBRATION.MODE == "Post-hoc":
        compute_metrics(cfg, test_data)

    #-----------Training model-----------------#
    if cfg.CALIBRATION.NEED_CLASSIFIER_TRAIN or cfg.CALIBRATION.NEED_CALIBRATION_TRAIN:
        DoTrain = doTrain(cfg)
        DoTrain.do_train(train_data,valid_data,test_data,model)

    #-----------Compute Metrics After Calibration-----------#
    compute_metrics(cfg,test_data,model=model,Final_record=True)

    ##---------plot----------------#
    pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    Specific_cfg_path = None
    main(Specific_cfg_path)