import torch
import os
import sys
import inspect

parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
sys.path.insert(0,parentdir) 
from customKing.utils.env import Set_seed
from customKing.config.config import get_cfg
from customKing.modeling.meta_arch.build import build_model, build_metric
from customKing.data.datasets.Public_logits_data.unpickle_probs import unpickle_probs
from customKing.data.datasets.Generate_logits_data.unpickle_generate_logits import unpickle_g_logits

import logging
from customKing.solver.build import build_optimizer,build_lr_scheduler
import time
import json
import torch.nn.functional as F
from visual.calibration_visual.Reliability_diagram import plot_reliability_diagram,plot_calibraiton_curve

class doTrain():
    def __init__(self,cfg):
        self.iteration = cfg.SOLVER.START_ITER
        self.cfg = cfg

    def do_train(self,train_set,valid_set,test_set,model):
        logging.basicConfig(level=logging.INFO)
        if model.require_iterative_training:
            model.train() 
            optimizer = torch.optim.SGD(model.parameters(),lr=self.cfg.SOLVER.BASE_LR,momentum=self.cfg.SOLVER.MOMENTUM,weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,nesterov=self.cfg.SOLVER.NESTEROV)
            scheduler = build_lr_scheduler(self.cfg.SOLVER, optimizer) 
            #---------Train------------#
            for epoch in range(self.cfg.SOLVER.MAX_EPOCH):
                if len(valid_set) == 2:
                    z_val, y_val = valid_set
                else:
                    feature_val,z_val, y_val = valid_set
                z_val,y_val = numpy_to_tensor(self.cfg,z_val,y_val)
                loss_value= model(z_val,y_val)
                optimizer.zero_grad()    
                loss_value.backward()    
                optimizer.step()

                with open(self.cfg.MODEL.OUTPUT_DIR+r"logging.json",mode="a") as f:
                    f.write(json.dumps({"Epoch":epoch,"loss_value":loss_value.item(), "lr": optimizer.param_groups[0]["lr"]})+"\n")

                #-----------Compute Metrics After Calibration-----------#
                compute_metrics(self.cfg,test_set,model=model)
                
                scheduler.step()
        else:
            if len(valid_set) == 2:
                z_val, y_val = valid_set
                z_val,y_val = numpy_to_tensor(self.cfg,z_val,y_val)
                model.train()
                model(z_val,y_val)
            elif len(valid_set) == 3:
                feature_val,z_val, y_val = valid_set
                feature_train,z_train,y_train = train_set
                model.train()
                signature = inspect.signature(model.forward)
                names = []
                for name, param in signature.parameters.items():
                    names.append(name)
                if len(names) == 5:
                    model(z_val,y_val,feature_val,feature_train,y_train)
                elif len(names) == 4:
                    model(z_val,y_val,z_train,y_train)
                elif len(names) == 2:
                    model(z_val,y_val)

def plot(self,OUTPUT_DIR, Dataset, n_bins=15, model = None):
    z_list, label_list = Dataset

    z_list = torch.stack(z_list,dim=0).to(self.cfg.MODEL.DEVICE)
    label_list = torch.stack(label_list,dim=0).to(self.cfg.MODEL.DEVICE)
    after = False
    if model != None:
        model.eval()
        cali_confidence,hits,is_softmaxed = model(z_list,label_list)
    
    plot_calibraiton_curve(cali_confidence, hits, OUTPUT_DIR)

def numpy_to_tensor(cfg,z,y):
    z = torch.from_numpy(z)
    y = torch.from_numpy(y)
    if len(y.shape) == 2:
        y = y.squeeze(dim=1)
    z = z.to(cfg.MODEL.DEVICE)
    y = y.to(cfg.MODEL.DEVICE).long()
    return z,y

def logit_to_confidence(z,y):
    softmaxes = F.softmax(z, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    confidences,resort_index = torch.sort(confidences)
    labels = y[resort_index]
    predictions = predictions[resort_index]
    return confidences,predictions,labels

def compute_metrics(cfg, Dataset, model=None, Final_record = False):
        if len(Dataset) == 2:
            z, y = Dataset
        elif len(Dataset) == 3:
            feature,z, y = Dataset

        if model is not None:
            model.eval()
            cali_confidence,predictions,labels = model(z,y)   # tensor
            hits = predictions.eq(labels)
            cali_confidence = cali_confidence.cpu().detach().numpy()
            hits = hits.cpu().detach().numpy().astype(int)
            labels = labels.cpu().detach().numpy().astype(int)

            model.train()
            metric_values = {}
            for metric in cfg.EVALUATE.METHOD_list:
                metric_method = build_metric(cfg,metric)
                signature = inspect.signature(metric_method.forward)
                names = []
                for name, param in signature.parameters.items():
                    names.append(name)
                if len(names)==2:
                    metric_value = metric_method(cali_confidence, hits)
                else:
                    metric_value = metric_method(cali_confidence,hits,labels)
                metric_values[metric] = metric_value
        else:
            z,y = numpy_to_tensor(cfg,z,y)
            confidence,predictions,labels = logit_to_confidence(z,y)
            hits = predictions.eq(labels)
            confidence = confidence.cpu().detach().numpy()
            hits = hits.cpu().detach().numpy().astype(int)
            labels = labels.cpu().detach().numpy().astype(int)
            metric_values = {}
            for metric in cfg.EVALUATE.METHOD_list:
                metric_method = build_metric(cfg,metric)
                signature = inspect.signature(metric_method.forward)
                names = []
                for name, param in signature.parameters.items():
                    names.append(name)
                if len(names)==2:
                    metric_value = metric_method(confidence, hits)
                else:
                    metric_value = metric_method(confidence,hits,labels)
                metric_values[metric] = metric_value

        if Final_record:
            with open(cfg.MODEL.OUTPUT_DIR+r"Final_logging.json",mode="a") as f:
                f.write(json.dumps(metric_values)+"\n")
                if hasattr(model,"temperature"):
                    f.write(json.dumps({"temperature":model.temperature.item()}))
        else:
            with open(cfg.MODEL.OUTPUT_DIR+r"logging.json",mode="a") as f:
                if model is None:
                    f.write("Uncalibration:"+"\n")
                f.write(json.dumps(metric_values)+"\n")

def main():
    
    #-----------Load config----------------#
    Set_seed(seed=20)
    task_mode = "Post_Calibration"
    cfg = get_cfg(task_mode)
    if not os.path.exists(cfg.MODEL.OUTPUT_DIR):
        os.makedirs(cfg.MODEL.OUTPUT_DIR)
    #save cfg
    if os.path.exists(os.path.join(cfg.MODEL.OUTPUT_DIR,'cfg.yaml')):
        os.remove(os.path.join(cfg.MODEL.OUTPUT_DIR,'cfg.yaml'))
    with open(os.path.join(cfg.MODEL.OUTPUT_DIR,'cfg.yaml'), 'a+') as f:
        print(cfg, file=f)

    #-----------Load Data------------------#
    _root = os.path.expanduser(os.getenv("CUSTOM_KING_DATASETS", "datasets"))
    data_path = os.path.join(_root,cfg.DATASET.DATA_PATH)
    if "public_logits_datasets" in data_path:
        valid_set, test_set = unpickle_probs(data_path, False)
    else:
        train_set, valid_set, test_set = unpickle_g_logits(data_path,False)

    #-----------Load model-----------------#
    if cfg.MODEL.PRE_WEIGHT:
        model = torch.load(cfg.MODEL.PREWEIGHT)
    else:
        model = build_model(cfg,None,None)
    logging.info(f"Calibration method:{cfg.MODEL.META_ARCHITECTURE}")
    logging.info("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    model.to(cfg.MODEL.DEVICE)

    #-----------Compute Metrics Before Calibration-----------#
    compute_metrics(cfg, test_set)

    if model.need_calibration_train == True:
        DoTrain = doTrain(cfg)
        DoTrain.do_train(train_set,valid_set,test_set,model)

    #-----------Compute Metrics After Calibration-----------#
    compute_metrics(cfg,test_set,model=model,Final_record=True)

    ##---------plot----------------#
    #plot_reliability_diagram(cfg.MODEL.OUTPUT_DIR+"reliability_diagram/after_calibration",test_set,n_bins=15, model=model)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()