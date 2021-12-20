import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


DLEN = 0

parser = argparse.ArgumentParser(description='csv analystic')
parser.add_argument('--csvDir', default=None)


def getAccIndex(rows):
    r_data = np.array(rows[1:], dtype=np.float32)
    acc_index = r_data[:, -1] == 1
    y = r_data[acc_index, 0]
    x = r_data[acc_index, 1]

    return x, y


def getAccPerClass(rows):
    global DLEN
    data_num = len(rows)
    DLEN = data_num
    r_data = np.array(rows[1:], dtype=np.float32)
    class_sets = list(set(r_data[:, -2].tolist()))
    all_acc = r_data[r_data[:, -1]==1.0, 0].shape[0] / data_num
    
    acc = []
    for c in class_sets:
        c_array = r_data[r_data[:,-2]==c,:]
        acc_c_index = c_array[c_array[:,-1]==1.0, :]
        acc.append(acc_c_index.shape[0]/data_num)

    # whole accuracy
    class_sets.append(-1.0)
    acc.append(all_acc)

    return class_sets, acc


def EachImage(args):
    all_files = os.listdir(args.csvDir)
    csv_files = [f for f in all_files if "result_per_image.csv" in f]
    file_num = len(csv_files)
    csv_files.sort()

    fig = plt.figure(figsize=(40, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    
    PerImage_x_list = []
    PerImage_y_list = []

    for f in csv_files:
        with open(os.path.join(args.csvDir, f)) as rf:
            reader = csv.reader(rf)
            rows = [row for row in reader]

        class_idx, acc = getAccPerClass(rows=rows[1:])
        ax2.scatter(class_idx, acc, label="{:.1f}".format(float(rows[1][0])))
        x, y = getAccIndex(rows=rows[1:])
        PerImage_x_list.append(x)
        PerImage_y_list.append(y)
        print("File {}, Acc {}".format(f, acc[-1]))
    
    PerImage_x = np.concatenate(PerImage_x_list)
    PerImage_y = np.concatenate(PerImage_y_list)
    
    # Accurate plot per image
    ax1.scatter(PerImage_x, PerImage_y, s=0.5)
    ax2.set_xticks(class_idx)
    ax2.legend()

    savefile = os.path.join(args.csvDir, "Acc_Analay_Plot.png")
    fig.savefig(savefile, dpi=150)

def EachTime(args):
    all_files = os.listdir(args.csvDir)
    csv_files = [f for f in all_files if "batchAcc_per_time.csv" in f]
    file_num = len(csv_files)
    csv_files.sort()

    fig = plt.figure(figsize=(6, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    all_acc_x = []
    all_acc_y = []

    for f in csv_files:
        vth = f.replace("Vth-", "")
        vth = vth.replace("_batchAcc_per_time.csv", "")
        if vth == "Dynamic":
            all_acc_x.append(-1)
        else:
            all_acc_x.append(float(vth))

        with open(os.path.join(args.csvDir, f)) as rf:
            reader = csv.reader(rf)
            rows = [list(map(int, row)) for row in reader]
    
        acc_batch_step = np.array(rows)
        timestep = acc_batch_step.shape[1]
        acc_batch_step = np.sum(acc_batch_step, axis=0) / DLEN
        x = np.arange(timestep+1)
        y = np.pad(acc_batch_step, ((1, 0)))
        ax1.plot(x, y, label="{}".format(vth))
        all_acc_y.append(y[-1])
    
    ax1.legend()
    ax2.scatter(all_acc_x, all_acc_y)
    savefile = os.path.join(args.csvDir, "Acc_TimeStep_Plot.png")
    fig.savefig(savefile, dpi=150)

if __name__ == "__main__":
    args = parser.parse_args()
    EachImage(args=args)
    EachTime(args=args)
