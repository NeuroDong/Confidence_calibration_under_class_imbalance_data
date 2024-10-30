import matplotlib.pyplot as plt
import numpy as np

def plot_Quartile_chart(x_list,y_list_list):
    fig = plt.figure()
    # boxplot
    boxprops = dict(linestyle='-', linewidth=3, color='red')
    whiskerprops = dict(linestyle='--', linewidth=3, color='green')
    capprops = dict(linestyle='-', linewidth=3, color='blue')
    medianprops = dict(linestyle='-', linewidth=3, color='black')
    flierprops=dict(marker='o', markersize=10)
    plt.boxplot(y_list_list,positions=x_list,widths=0.02,boxprops=boxprops,whiskerprops=whiskerprops, capprops=capprops,flierprops=flierprops, medianprops=medianprops,patch_artist=True)

    plt.xlabel("Confidence")
    plt.ylabel("Mean accuracy")
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend(ncol=2)

def plot_Variance_Bands(x_list,y_list_list):
    # Internl plot
    fig = plt.figure()
    mean = np.mean(y_list_list,axis=1)
    std = np.std(y_list_list,axis=1)
    plt.plot(x_list, mean, label='Mean', color='blue')
    # 绘制数据带
    plt.fill_between(x_list, mean - std, mean + std, color='blue', alpha=0.2, label='Mean ± Std')
    # 添加标题和标签
    plt.title('Data Band with Mean and Std')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.ylim([0,1])

def plot_Max_Min_Bands(x_list,y_list_list):
    # Internl plot
    #fig = plt.figure()
    mean = np.mean(y_list_list,axis=1)
    min_line = np.min(y_list_list,axis=1)
    max_line = np.max(y_list_list,axis=1)
    index = [max_line[i]!=0 for i in range(len(max_line))]
    x_list = np.array(x_list)
    x_list = x_list[index]
    mean = mean[index]
    min_line = min_line[index]
    max_line = max_line[index]

    plt.plot(x_list, mean, linewidth = 4, label='HB\'s mean result (bins from 10 to 50)', color='green')
    # 绘制数据带
    plt.fill_between(x_list, min_line, max_line, color='blue', alpha=0.2, label='HB\'s result range (bins from 10 to 50)')

    plt.xlabel('Confidence',fontname="Times New Roman",fontsize=30)
    #plt.ylabel('Accuracy',fontname="Times New Roman",fontsize=30)
    plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)
    ax = plt.gca()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)  # 设置线条宽度
        ax.spines[axis].set_color('black')  # 设置线条颜色

    plt.legend(prop={"family": "Times New Roman","size":30},loc="upper left")
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.tick_params(axis='both', labelsize=30)
    return x_list,mean