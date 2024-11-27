import torch
import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
sys.path.insert(0,parentdir) 
from customKing.utils.env import Set_seed
from customKing.config.config import get_cfg
from customKing.modeling.meta_arch.build import build_model, build_metric
from customKing.data.datasets.Public_logits_data.unpickle_probs import unpickle_probs

import logging
from customKing.solver.build import build_optimizer,build_lr_scheduler
import time
import json
import torch.nn.functional as F
from visual.calibration_visual.Reliability_diagram import plot_reliability_diagram,plot_calibraiton_curve

from customKing.data import get_dataset_dicts,build_loader 
from tqdm import tqdm


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
            if model.output_uncertainty:
                inference_result,uncertainty = model(batch_img,batch_label)
                logit_list.append(inference_result)
                inference_result = F.softmax(inference_result, dim=1)
                _, result = torch.max(inference_result.data, 1)
                result_list.append(result)
                label_list.append(batch_label)
                confidence_list.append(uncertainty)
            else:
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

    def do_train(self,train_data,test_data,model):
        logging.basicConfig(level=logging.INFO)
        model.train() 
        optimizer = torch.optim.SGD(model.parameters(),lr=self.cfg.SOLVER.BASE_LR,momentum=self.cfg.SOLVER.MOMENTUM,weight_decay=self.cfg.SOLVER.WEIGHT_DECAY,nesterov=self.cfg.SOLVER.NESTEROV)
        scheduler = build_lr_scheduler(self.cfg.SOLVER, optimizer)
        
        #---------Train------------#
        if model.Two_stage:
            stages = [1,2]
        else:
            stages = [1]

        for stage in stages:
            for epoch in range(self.cfg.SOLVER.MAX_EPOCH):
                if self.iteration > self.cfg.SOLVER.MAX_ITER:
                    break
                time1 = time.time()
                if hasattr(model,"classifier"):
                    if hasattr(model.classifier, '_hook_before_iter'):
                        model.classifier._hook_before_iter()
                if hasattr(model, '_hook_before_epoch'):
                    model._hook_before_epoch(epoch)
                log_iteration = len(train_data)-1 if (len(train_data) < 20) else 20
                for batch_img,batch_label in train_data:
                    self.iteration = self.iteration + 1
                    batch_img = batch_img.cuda().clone().detach().float()
                    batch_label = batch_label.cuda().float().long()
                    if model.Two_stage:
                        predict,losses = model(batch_img,batch_label,stage)
                    else:
                        if model.need_epoch:
                            predict,losses = model(batch_img,batch_label,epoch)
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
                    if self.iteration % log_iteration == 0:
                        with open(self.cfg.MODEL.OUTPUT_DIR+r"logging.json",mode="a") as f:
                            f.write(json.dumps({"Stage":stage, "Epoch":epoch, "Iteration":self.iteration, "train_acc":train_acc.item(), "loss_value":loss.item(), "lr": optimizer.param_groups[0]["lr"]})+"\n")
                #-----------------test model----------------#
                compute_metrics(self.cfg,test_data,model=model)

def main():
    
    #-----------Load config----------------#
    Set_seed(seed=20)
    task_mode = "In-Train Calibration"
    cfg = get_cfg(task_mode)
    if not os.path.exists(cfg.MODEL.OUTPUT_DIR):
        os.makedirs(cfg.MODEL.OUTPUT_DIR)
    #save cfg
    if os.path.exists(os.path.join(cfg.MODEL.OUTPUT_DIR,'cfg.yaml')):
        os.remove(os.path.join(cfg.MODEL.OUTPUT_DIR,'cfg.yaml'))
    with open(os.path.join(cfg.MODEL.OUTPUT_DIR,'cfg.yaml'), 'a+') as f:
        print(cfg, file=f)

    #-----------Load Data------------------#
    train_data = build_loader(cfg, get_dataset_dicts(cfg.DATASETS.TRAIN))
    cls_num_list_train = train_data.dataset.get_cls_num_list()
    test_data = build_loader(cfg, get_dataset_dicts(cfg.DATASETS.TEST))
    cls_num_list_test = test_data.dataset.get_cls_num_list()

    #-----------Load model-----------------#
    if cfg.MODEL.PRE_WEIGHT:
        model = torch.load(cfg.MODEL.PREWEIGHT)
    else:
        model = build_model(cfg,cls_num_list_train,cls_num_list_test)
    logging.info("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))
    model.to(cfg.MODEL.DEVICE)

    DoTrain = doTrain(cfg)
    DoTrain.do_train(train_data,test_data,model)

    #-----------------test model----------------#
    compute_metrics(cfg,test_data,model=model,Final_record=True)

    ##---------plot----------------#
    #plot_reliability_diagram(cfg.MODEL.OUTPUT_DIR+"reliability_diagram/after_calibration",test_data,n_bins=15, model=model)

if __name__ == "__main__":
    main()