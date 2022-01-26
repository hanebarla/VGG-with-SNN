import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import is_num


DLEN = 0
BURNIN = 0

parser = argparse.ArgumentParser(description='csv analystic')
parser.add_argument('--csvDir', default=None)
parser.add_argument('--Fire', default=0, type=int)


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
    vth_files = [f for f in all_files if "Vth_per_time.csv" in f]

    file_num = len(csv_files)
    csv_files.sort()

    fig = plt.figure(figsize=(6, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for f in csv_files:
        vth = f.replace("Vth-", "")
        vth = vth.replace("_batchAcc_per_time.csv", "")
        # alpha = float(vth.replace("Dynamic_alpha-", ""))

        with open(os.path.join(args.csvDir, f)) as rf:
            reader = csv.reader(rf)
            rows = [list(map(int, row)) for row in reader]
    
        acc_batch_step = np.array(rows)
        timestep = acc_batch_step.shape[1]
        acc_batch_step = np.sum(acc_batch_step, axis=0) / DLEN
        print(f)
        if "burnin" in f:
            idx = vth.find("burnin-")
            bunrin_time = int(vth[idx+7:])
            global BURNIN
            BURNIN = bunrin_time
            timestep += bunrin_time-1
            y = np.pad(acc_batch_step, (bunrin_time, 0))
        else:
            y = np.pad(acc_batch_step, ((1, 0)))
        x = np.arange(timestep+1)
        ax1.plot(x, y, label="{}".format(vth))
        print("{}: {}".format(vth, y[-1]))

        if is_num(vth):
            const_vth = f.replace("Vth-", "")
            const_vth = float(const_vth.replace("_batchAcc_per_time.csv", ""))
            Vth_y = np.full_like(x, const_vth)
            ax2.plot(x, Vth_y, label="{}".format(vth))

    for f in vth_files:
        if "burnin" in f:
            break

        vth = f.replace("Vth-", "")
        vth = vth.replace("_Vth_per_time.csv", "")

        with open(os.path.join(args.csvDir, f)) as rf:
            reader = csv.reader(rf)
            rows = [list(map(float, row)) for row in reader]

        Vthnp = np.array(rows)
        Vth_y = np.mean(rows, axis=0)
        if "Dynamic" in f:
            ax2.plot(x, Vth_y, label="{}".format(vth))
            # ax2.set_xlim([0, 100])
        elif "Adaptive" in f:
            ax2.plot(x, Vth_y, label="{}".format(vth))
        else:
            raise ValueError
    
    ax1.legend()
    ax2.legend()
    savefile = os.path.join(args.csvDir, "Acc_TimeStep_Plot.png")
    fig.savefig(savefile, dpi=150)

def FireTime(args):
    all_files = os.listdir(args.csvDir)
    csv_files = [f for f in all_files if "Firecount_per_time.csv" in f]
    file_num = len(csv_files)
    csv_files.sort()

    cm = plt.cm.get_cmap("tab20")
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(111)
    
    for f in csv_files:
        with open(os.path.join(args.csvDir, f)) as rf:
            reader = csv.reader(rf)
            rows = [list(map(float, row)) for row in reader]
            rows = np.array(rows)

        layer_num = rows.shape[1] - 1
        for l in range(layer_num):
            ax1.scatter(rows[:,0], rows[:,l+1]/(rows[:,0]+1), label="{}-Layer".format(l+1), color=cm(l))
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        savefile = os.path.join(args.csvDir, "{}_FireCount_TimeStep_Plot.png".format(f.replace("_Firecount_per_time.csv", "")))
        fig.savefig(savefile, dpi=150)

def Energy(args):
    all_files = os.listdir(args.csvDir)
    csv_files = [f for f in all_files if "Energy_per_time.csv" in f]
    csv_files.sort()

    fig = plt.figure(figsize=(6, 8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for f in csv_files:
        if "burnn" in f:
            continue
        with open(os.path.join(args.csvDir, f)) as rf:
            reader = csv.reader(rf)
            rows = [list(map(float, row)) for row in reader]
            rows = np.array(rows)
            rows = np.pad(rows, ((0, 0), (1, 0)))
        timestep = rows.shape[1]
        rows *= 1e-12
        x = np.arange(timestep)

        vth_in_file = f.replace("_Energy_per_time.csv", "")
        vth_in_file = vth_in_file.replace("Vth-", "")
        if is_num(vth_in_file):
            rows /= float(vth_in_file)
        else:
            with open(os.path.join(args.csvDir, f.replace("Energy", "Vth"))) as rf:
                reader = csv.reader(rf)
                Vth_rows = [list(map(float, row)) for row in reader]
                Vth_rows = np.array(Vth_rows)
                rows /= Vth_rows
        rows = np.sum(rows, axis=0)
        rows = np.cumsum(rows)
        ax1.scatter(x, rows, label="{}".format(str(f).replace("_Energy_per_time.csv", "")), s=0.1)

        with open(os.path.join(args.csvDir, str(f).replace("Energy", "batchAcc"))) as rf:
            reader = csv.reader(rf)
            acc_rows = [list(map(int, row)) for row in reader]
        
        acc_batch_step = np.sum(acc_rows, axis=0) / DLEN
        acc_batch_step = np.pad(acc_batch_step, ((1,0)))
        ax2.scatter(rows, acc_batch_step, label="{}".format(str(f).replace("_Energy_per_time.csv", "")), s=0.1)

    ax1.legend()
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Energy [T*fire]")
    ax2.legend()
    ax2.set_xlabel("Energy [T*fire]")
    ax2.set_ylabel("Accuracy")

    savefile = os.path.join(args.csvDir, "Energy_Plot.png")
    fig.savefig(savefile, dpi=150)

if __name__ == "__main__":
    args = parser.parse_args()
    EachImage(args=args)
    EachTime(args=args)
    if args.Fire == 1:
        FireTime(args=args)
    Energy(args=args)
