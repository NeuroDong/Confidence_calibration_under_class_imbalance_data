import torch
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
sys.path.insert(0,parentdir) 
from customKing.utils.env import Set_seed
from customKing.config.config import get_cfg
from customKing.modeling.meta_arch.build import build_model, build_metric,META_ARCH_REGISTRY
from customKing.data.datasets.Public_logits_data.unpickle_probs import unpickle_probs

import logging
from customKing.solver.build import build_optimizer,build_lr_scheduler
import time
import json
import torch.nn.functional as F
from visual.calibration_visual.Reliability_diagram import plot_reliability_diagram,plot_calibraiton_curve

from customKing.data import get_dataset_dicts,build_loader,build_sampling_loader
from tqdm import tqdm
import numpy as np

def compute_test_acc(cfg, Dataset, model):
    model.eval()
    result_list = []
    label_list = []
    confidence_list = []
    logit_list = []
    with torch.no_grad():
        for batch_img,batch_label in tqdm(Dataset,dynamic_ncols=True):
            batch_label = batch_label.cuda().clone().detach().float()
            batch_img = batch_img.cuda().clone().detach().float()  
            inference_result = model(batch_img,batch_label)
            logit_list.append(inference_result)
            inference_result = F.softmax(inference_result, dim=1)
            confidence, result = torch.max(inference_result.data, 1)
            result_list.append(result)
            label_list.append(batch_label)
            confidence_list.append(confidence)

    logits = torch.cat(logit_list,dim=0)
    results = torch.cat(result_list,dim=0)
    labels = torch.cat(label_list,dim=0)
    confidences = torch.cat(confidence_list,dim=0)
    hits = results.eq(labels)

    #------------compute Acc------------#
    acc = sum(hits)/len(hits)
    with open(cfg.MODEL.OUTPUT_DIR+r"logging.json",mode="a") as f:
        f.write(json.dumps({"testAcc": acc.item()})+"\n")

    return logits,confidences,labels,hits,acc

def compute_metrics(cfg, Dataset, model=None, Final_record = False):

    logits,confidences,labels,hits,acc = compute_test_acc(cfg, Dataset, model)
    confidences,resort_index = torch.sort(confidences)
    logits = logits[resort_index]
    labels = labels[resort_index]
    hits = hits[resort_index]
    confidences = confidences.cpu().numpy()
    hits = hits.cpu().numpy().astype(int)

    #-----------compute calibration metrics---------------#
    metric_values = {}
    for metric in cfg.EVALUATE.METHOD_list:
        metric_method = build_metric(cfg,metric)
        if metric_method.Need_top_confidence:
            metric_value = metric_method(confidences, hits)
        else:
            metric_value = metric_method(logits,labels)
        metric_values[metric] = metric_value


    if Final_record:
        with open(cfg.MODEL.OUTPUT_DIR+r"Final_logging.json",mode="a") as f:
            f.write(json.dumps(metric_values)+"\n")
            if hasattr(model,"temperature"):
                f.write(json.dumps({"temperature":model.temperature.item()}))
    else:
        with open(cfg.MODEL.OUTPUT_DIR+r"logging.json",mode="a") as f:
            f.write(json.dumps(metric_values)+"\n")

    model.train()

class doTrain():
    def __init__(self,cfg):
        self.iteration = cfg.SOLVER.START_ITER
        self.cfg = cfg

    def do_train(self,train_data,valid_data,test_data,model):
        logging.basicConfig(level=logging.INFO)
        optimizer = torch.optim.SGD(model.parameters(),lr=self.cfg.SOLVER.BASE_LR,momentum=self.cfg.SOLVER.MOMENTUM,weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,nesterov=self.cfg.SOLVER.NESTEROV)
        scheduler = build_lr_scheduler(self.cfg.SOLVER, optimizer) 
        
        #---------Train------------#
        # Train classifier
        for epoch in range(self.cfg.SOLVER.MAX_EPOCH):
            if self.iteration > self.cfg.SOLVER.MAX_ITER:
                break
            time1 = time.time()
            log_iteration = len(train_data) if (len(train_data) < 20) else 20

            for batch_img,batch_label in train_data:
                model.train()
                self.iteration = self.iteration + 1

                batch_img = batch_img.cuda().clone().detach().float()    
                batch_label = batch_label.cuda().float().long()           

                #data augmentation
                if self.cfg.MODEL.AUGMENTATION == "unimix":
                    aug_model = META_ARCH_REGISTRY.get(self.cfg.MODEL.AUGMENTATION)(self.cfg)
                    train_cls_num_list = np.array(train_data.dataset.get_cls_num_list())
                    mixed_images, labels_1, labels_2, lam = aug_model(batch_img,batch_label,train_cls_num_list)
                    test_cls_num_list = np.array(test_data.dataset.get_cls_num_list())
                    criterion = META_ARCH_REGISTRY.get("Bayias_compensated_loss")(train_cls_num_list,test_cls_num_list)
                    predict = model._forward_impl(mixed_images)
                    losses = torch.mean(lam * criterion(predict, labels_1) + (1 - lam) * criterion(predict, labels_2))
                else:
                    predict,losses = model(batch_img,batch_label)  

                time2 = time.time()
                optimizer.zero_grad()   
                if self.cfg.SOLVER.IS_PARALLEL:
                    loss = losses.sum()/len(losses)
                    loss.backward()
                else:
                    loss = losses
                    loss.backward()   
                optimizer.step() 

                #---------Calculate the training accuracy (that is, the accuracy within a batch)-----#
                _, predicted = torch.max(predict.data, 1)
                correct = predicted.eq(batch_label).cpu().sum()
                train_acc = correct / batch_img.shape[0]

                scheduler.step()

                if self.iteration % log_iteration ==0:
                    with open(self.cfg.MODEL.OUTPUT_DIR+r"logging.json",mode="a") as f:
                        f.write(json.dumps({"Epoch":epoch, "Iteration":self.iteration, "train_acc":train_acc.item(), "loss_value":loss.item(), "lr": optimizer.param_groups[0]["lr"]})+"\n")

            compute_test_acc(self.cfg, test_data, model)

        #Train calibration model
        if self.cfg.MODEL.NEED_CALIBRATION_TRAIN:
            model.eval()
            meta_arch = self.cfg.MODEL.CALIBRATION_METHOD
            calibration_model = META_ARCH_REGISTRY.get(meta_arch)(self.cfg,train_data.sampler.beta)

            if calibration_model.need_calibration_train == True:
                if calibration_model.require_iterative_training:
                    pass
                else:
                    z_val = []
                    y_val = []
                    for batch_img,batch_label in valid_data:
                        batch_img = batch_img.cuda().clone().detach().float()    
                        batch_label = batch_label.cuda().float().long()                        
                        predict = model(batch_img,batch_label) 
                        z_val.append(predict)
                        y_val.append(batch_label)

                    logits_val = torch.cat(z_val,dim=0)
                    labels_val = torch.cat(y_val,dim=0)
                    calibration_model.train()
                    calibration_model(logits_val,labels_val)

        #-----------------test model----------------#
        compute_metrics(self.cfg,test_data,model=model)

def main():
    
    #-----------Load config----------------#
    Set_seed(seed=20)
    task_mode = "DataLevel Calibration"
    cfg = get_cfg(task_mode)
    if not os.path.exists(cfg.MODEL.OUTPUT_DIR):
        os.makedirs(cfg.MODEL.OUTPUT_DIR)
    #save cfg
    if os.path.exists(os.path.join(cfg.MODEL.OUTPUT_DIR,'cfg.yaml')):
        os.remove(os.path.join(cfg.MODEL.OUTPUT_DIR,'cfg.yaml'))
    with open(os.path.join(cfg.MODEL.OUTPUT_DIR,'cfg.yaml'), 'a+') as f:
        print(cfg, file=f)


    #-----------Load Data------------------#
    train_data = build_sampling_loader(cfg, get_dataset_dicts(cfg.DATASETS.TRAIN))
    valid_data = build_loader(cfg, get_dataset_dicts(cfg.DATASETS.VALID))
    test_data = build_loader(cfg, get_dataset_dicts(cfg.DATASETS.TEST))

    #-----------Load model-----------------#
    if cfg.MODEL.PRE_WEIGHT:
        model = torch.load(cfg.MODEL.PREWEIGHT)
    else:
        model = build_model(cfg)
    logging.info("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    model.to(cfg.MODEL.DEVICE)

    DoTrain = doTrain(cfg)
    DoTrain.do_train(train_data,valid_data,test_data,model)

    #-----------------test model----------------#
    compute_metrics(cfg,test_data,model=model,Final_record=True)

    ##---------plot----------------#
    #plot_reliability_diagram(cfg.MODEL.OUTPUT_DIR+"reliability_diagram/after_calibration",test_data,n_bins=15, model=model)

if __name__ == "__main__":
    main()