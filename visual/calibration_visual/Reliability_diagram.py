import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import seaborn as sns
import torch

def plot_calibraiton_curve(confidence, hits, output_dir, n_bins=15, after = True):
    '''
    Confidence: top-label value of softmax output
    hits: 0 or 1, 1 represents "hit"
    output_dir: the folder path of ploted image
    n_bins: the number of binning
    after: True if calibrated, False otherwise.
    '''
    
    fraction_of_positives, mean_predicted_value = calibration_curve(hits, confidence, n_bins=n_bins, strategy="uniform")

    plt.plot([0, 1], [0, 1], "k:",linewidth = 10)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth = 10)
    fontsize = 40    
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    plt.ylabel("Accuracy",fontsize=fontsize,fontname="Times New Roman")
    plt.xlabel("Confidence",fontsize=fontsize,fontname="Times New Roman")
    plt.ylim([0.4,1])
    plt.xlim([0.4,1])
    plt.tick_params(axis='both', labelsize=fontsize)

    #ax.legend(prop={"family": "Times New Roman","size":40},framealpha=0.1,loc="upper left")
    # 保存图像
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if after:
        plt.savefig(os.path.join(output_dir, 'after_cali_curve.eps'))
        plt.savefig(os.path.join(output_dir, 'after_cali_curve.png'))
    else:
        plt.savefig(os.path.join(output_dir, 'before_cali_curve.eps'))
        plt.savefig(os.path.join(output_dir, 'before_cali_curve.png'))
    plt.close()

def plot_reliability_diagram(confidence, hits, output_dir, n_bins=15, after = True):
    '''
    Confidence: top-label value of softmax output
    hits: 0 or 1, 1 represents "hit"
    output_dir: the folder path of ploted image
    n_bins: the number of binning
    after: True if calibrated, False otherwise.
    '''
    pass




def plot_reliability_diagram_for_multiclass(cfg,y_test, prob_pos_clf, output_dir, n_bins=15, after = False):
    class_labels = np.unique(y_test)
    if "Top_label" in cfg.MODEL.META_ARCHITECTURES[0]:
        top_labels = np.argmax(prob_pos_clf, axis=1)

    class_labels= class_labels[1::2]

    fig, axs = plt.subplots(1, len(class_labels), figsize=(12 * len(class_labels), 12))
    plt.subplots_adjust(wspace=0.5,bottom=0.3)
    fontsize = 80

    for i in range(len(class_labels)):
        ax = axs[i]
        if "Top_label" in cfg.MODEL.META_ARCHITECTURES[0]:
            class_index = (top_labels==class_labels[i])
            prob_pos_clf_class = prob_pos_clf[class_index]
            y_test_class = y_test[class_index]
            confidence = np.max(prob_pos_clf_class,axis=1)
            predict = np.argmax(prob_pos_clf_class,axis=1)
            true_label = (y_test_class==predict)
        else:
            class_index = (y_test == class_labels[i])
            prob_pos_clf_class = prob_pos_clf[class_index]
            y_test_class = y_test[class_index]
            confidence = np.max(prob_pos_clf_class,axis=1)
            predict = np.argmax(prob_pos_clf_class,axis=1)
            true_label = (y_test_class==predict)
        fraction_of_positives, mean_predicted_value = calibration_curve(true_label, confidence, n_bins=n_bins, strategy="uniform")
        
        bin_width = 1.0 / n_bins
        bins = np.linspace(0, 1.0, n_bins + 1)
        x=[]
        y=[]
        gaps = []
        for j in range(n_bins):
            x.append(bins[j])
            if np.any((mean_predicted_value >= bins[j]) & (mean_predicted_value <= bins[j+1])):
                for k in range(len(fraction_of_positives)):
                    if (mean_predicted_value[k] >= bins[j]) & (mean_predicted_value[k] <= bins[j+1]):
                        y.append(fraction_of_positives[k])
                        gaps.append(fraction_of_positives[k]-bins[j])
            else:
                y.append(0.)
                gaps.append(0.)

        # ax_hist = axs[0, i]
        # sns.kdeplot(confidence, ax=ax_hist, fill=True)
        ax.plot([0, 1], [0, 1], "k:",linewidth = 10)
        ax.bar(x, y, width=bin_width, linewidth = 2,align='edge', alpha=0.8, edgecolor='black',label="Outputs")
        ax.bar(x, gaps, width=bin_width, linewidth = 2 , hatch="/s",align='edge', alpha=0.8, color='#FAD7A0', edgecolor='black',bottom=x,label="Gap")
        
        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        ax.set_ylabel("Accuracy",fontsize=fontsize,fontname="Times New Roman")
        ax.set_xlabel("Confidence",fontsize=fontsize,fontname="Times New Roman")
        ax.set_ylim([0.,1])
        ax.set_xlim([0.,1])
        ax.set_title("Class {}".format(int(class_labels[i])),fontsize=fontsize,fontname="Times New Roman")
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.legend(prop={"family": "Times New Roman","size":40},framealpha=0.1,loc="upper left")
    # 保存图像
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if after:
        plt.savefig(os.path.join(output_dir, 'after_cali_reliability_diagram.eps'))
        plt.savefig(os.path.join(output_dir, 'after_cali_reliability_diagram.png'))
    else:
        plt.savefig(os.path.join(output_dir, 'before_cali_reliability_diagram.eps'))
        plt.savefig(os.path.join(output_dir, 'before_cali_reliability_diagram.png'))
    plt.close()

def plot_calibraiton_curve_for_multiclass(cfg,y_test, prob_pos_clf, output_dir, n_bins=15, after = False):
    class_labels = np.unique(y_test)
    if "Top_label" in cfg.MODEL.META_ARCHITECTURES[0]:
        top_labels = np.argmax(prob_pos_clf, axis=1)

    class_labels= class_labels[1::2]

    fig, axs = plt.subplots(1, len(class_labels), figsize=(12 * len(class_labels), 12))
    plt.subplots_adjust(wspace=0.5,bottom=0.3)
    fontsize = 80

    for i in range(len(class_labels)):
        ax = axs[i]
        if "Top_label" in cfg.MODEL.META_ARCHITECTURES[0]:
            class_index = (top_labels==class_labels[i])
            prob_pos_clf_class = prob_pos_clf[class_index]
            y_test_class = y_test[class_index]
            confidence = np.max(prob_pos_clf_class,axis=1)
            predict = np.argmax(prob_pos_clf_class,axis=1)
            true_label = (y_test_class==predict)
        else:
            class_index = (y_test == class_labels[i])
            prob_pos_clf_class = prob_pos_clf[class_index]
            y_test_class = y_test[class_index]
            confidence = np.max(prob_pos_clf_class,axis=1)
            predict = np.argmax(prob_pos_clf_class,axis=1)
            true_label = (y_test_class==predict)
        fraction_of_positives, mean_predicted_value = calibration_curve(true_label, confidence, n_bins=n_bins, strategy="quantile")

        ax.plot([0, 1], [0, 1], "k:",linewidth = 10)
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", linewidth = 10)
        
        ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
        ax.set_ylabel("Accuracy",fontsize=fontsize,fontname="Times New Roman")
        ax.set_xlabel("Confidence",fontsize=fontsize,fontname="Times New Roman")
        ax.set_ylim([0.4,1])
        ax.set_xlim([0.4,1])
        ax.set_title("Class {}".format(int(class_labels[i])),fontsize=fontsize,fontname="Times New Roman")
        ax.tick_params(axis='both', labelsize=fontsize)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(5)  # 设置线条宽度
            ax.spines[axis].set_color('black')  # 设置线条颜色

        #ax.legend(prop={"family": "Times New Roman","size":40},framealpha=0.1,loc="upper left")
    # 保存图像
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if after:
        plt.savefig(os.path.join(output_dir, 'after_cali_curve.eps'))
        plt.savefig(os.path.join(output_dir, 'after_cali_curve.png'))
    else:
        plt.savefig(os.path.join(output_dir, 'before_cali_curve.eps'))
        plt.savefig(os.path.join(output_dir, 'before_cali_curve.png'))
    plt.close()


if __name__=="__main__":
    # 加载数据
    X, y = datasets.make_classification(n_samples=100000, n_features=20,
                                        n_informative=2, n_redundant=10,
                                        random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99,
                                                        random_state=42)

    # 训练模型
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    prob_pos_clf = clf.predict_proba(X_test)[:, 1]