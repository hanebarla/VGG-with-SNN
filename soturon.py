import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import is_num

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["font.size"] = 14

DLEN = 0
BURNIN = 0
BASEACC = 0
BASEACCBURNIN = 0
# viewACCMAX = 0.63
# viewACCMIN = -0.03
viewACCMAX = 0.73
viewACCMIN = 0.5

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
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    plt.rcParams["svg.fonttype"] = "none"

    savefile = os.path.join(args.csvDir, "Acc_Analay_Plot.svg")
    fig.savefig(savefile)


def EachTime(args):
    global BASEACC
    global BASEACCBURNIN
    all_files = os.listdir(args.csvDir)
    csv_files = [f for f in all_files if "batchAcc_per_time.csv" in f]
    vth_files = [f for f in all_files if "Vth_per_time.csv" in f]

    file_num = len(csv_files)
    csv_files.sort()

    fig = plt.figure(figsize=(11, 13))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    accTimeX = []
    accTimeY = []

    for f in csv_files:
        cross_point = 0
        vth = f.replace("Vth-", "")
        vth = vth.replace("_batchAcc_per_time.csv", "")
        # alpha = float(vth.replace("Dynamic_alpha-", ""))

        with open(os.path.join(args.csvDir, f)) as rf:
            reader = csv.reader(rf)
            rows = [list(map(int, row)) for row in reader]
    
        acc_batch_step = np.array(rows)
        timestep = acc_batch_step.shape[1]
        acc_batch_step = np.sum(acc_batch_step, axis=0) / DLEN
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
        labelname = file2label(str(vth))
        ax1.plot(x, y, label="{}".format(labelname))
        print("{}: {}".format(vth, y[-1]))

        if "1.0" == str(vth):
            BASEACC = y[-1]
            ax1.axhline(BASEACC, linestyle="dashed", color='#1f77b4')
        elif "1.0_burnin-300" == str(vth):
            BASEACCBURNIN = y[-1]
            ax1.axhline(BASEACCBURNIN, linestyle="dotted", color='g')
        else:
            if "burnin" in f:
                timesteps = np.where(y>=BASEACCBURNIN)
                if len(timesteps[0]) != 0:
                    ax1.axvline(timesteps[0][0], ymax=((BASEACCBURNIN-viewACCMIN)/(viewACCMAX-viewACCMIN)), linestyle="dotted", c='g')
            else:
                timesteps = np.where(y>=BASEACC)
                if len(timesteps[0]) != 0:
                    ax1.axvline(timesteps[0][0], ymax=((BASEACC-viewACCMIN)/(viewACCMAX-viewACCMIN)), linestyle="dashed", color='#1f77b4')

        if is_num(vth):
            const_vth = f.replace("Vth-", "")
            const_vth = float(const_vth.replace("_batchAcc_per_time.csv", ""))
            Vth_y = np.full_like(x, const_vth)
            labelname = file2label(str(vth))
            ax2.plot(x, Vth_y, label="{}".format(labelname))

    for f in vth_files:
        if "burnin" in f:
            continue

        vth = f.replace("Vth-", "")
        vth = vth.replace("_Vth_per_time.csv", "")

        with open(os.path.join(args.csvDir, f)) as rf:
            reader = csv.reader(rf)
            rows = [list(map(float, row)) for row in reader]

        Vthnp = np.array(rows)
        Vth_y = np.mean(rows, axis=0)
        if "Dynamic" in f:
            labelname = file2label(str(vth))
            Vth_y[0] = 2
            ax2.plot(x, Vth_y, label="{}".format(labelname))
            # ax2.set_xlim([0, 100])
        elif "Adaptive" in f:
            labelname = file2label(str(vth))
            ax2.plot(x, Vth_y, label="{}".format(labelname))
        else:
            raise ValueError
    
    ax1.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=2)
    ax1.set_xlabel("TimeStep")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("(a)", y=-0.2)
    ax2.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center')
    ax2.set_xlabel("TimeStep")
    ax2.set_ylabel("Vth")
    #ax2.set_title("(b)", y=-0.2)
    ax1.set_xlim(300, 1030)
    ax1.set_ylim(viewACCMIN, viewACCMAX)
    fig.tight_layout()

    plt.rcParams["svg.fonttype"] = "none"

    savefile = os.path.join(args.csvDir, "Acc_TimeStep_Plot.svg")
    fig.savefig(savefile)

def FireTime(args):
    all_files = os.listdir(args.csvDir)
    csv_files = [f for f in all_files if "Firecount_per_time.csv" in f]
    file_num = len(csv_files)
    csv_files.sort()

    cm = plt.cm.get_cmap("tab20")
    
    for f in csv_files:
        fig = plt.figure(figsize=(6, 5))
        ax1 = fig.add_subplot(111)
        with open(os.path.join(args.csvDir, f)) as rf:
            reader = csv.reader(rf)
            rows = [list(map(float, row)) for row in reader]
            rows = np.array(rows)

        layer_num = rows.shape[1] - 1
        for l in range(layer_num):
            ax1.scatter(rows[:,0], rows[:,l+1]/(rows[:,0]+1), label="{}-Layer".format(l+1), color=cm(l))
        
        ax1.set_xlabel("TimeStep")
        ax1.set_ylabel("Average fire rate")
        ax1.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=4)
        plt.tight_layout()
        savefile = os.path.join(args.csvDir, "{}_FireCount_TimeStep_Plot.png".format(f.replace("_Firecount_per_time.csv", "")))
        fig.savefig(savefile, dpi=600)

def Energy(args):
    all_files = os.listdir(args.csvDir)
    csv_files = [f for f in all_files if "Energy_per_time.csv" in f]
    csv_files.sort()

    fig = plt.figure(figsize=(11, 9))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for f in csv_files:
        cutf = f.replace("Vth-", "")
        with open(os.path.join(args.csvDir, f)) as rf:
            reader = csv.reader(rf)
            rows = [list(map(float, row)) for row in reader]
            rows = np.array(rows)
            rows = np.pad(rows, ((0, 0), (1, 0)))
        timestep = rows.shape[1]
        rows *= 9e-13
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
        labelname = cutf.replace("_Energy_per_time.csv", "")
        labelname = file2label(labelname)
        ax1.plot(x, rows, label="{}".format(labelname))

        with open(os.path.join(args.csvDir, str(f).replace("Energy", "batchAcc"))) as rf:
            reader = csv.reader(rf)
            acc_rows = [list(map(int, row)) for row in reader]

            acc_batch_step = np.sum(acc_rows, axis=0) / DLEN
            acc_batch_step = np.pad(acc_batch_step, ((1,0)))
            labelname = cutf.replace("_Energy_per_time.csv", "")
            labelname = file2label(labelname)
            ax2.plot(rows, acc_batch_step, label="{}".format(labelname))
            timesteps = np.where(acc_batch_step>=BASEACC)
            if len(timesteps[0]) != 0 and ("Vth-1.0" not in f):
                ax2.axvline(rows[timesteps[0][0]], ymax=((BASEACC-viewACCMIN)/(viewACCMAX-viewACCMIN)), linestyle="dashed", color='#1f77b4')

        with open(os.path.join(args.csvDir, str(f).replace("Energy", "burnin-300_batchAcc"))) as rf:
            reader = csv.reader(rf)
            burnin_acc_rows = [list(map(int, row)) for row in reader]
            burnin_acc_batch_step = np.sum(burnin_acc_rows, axis=0) / DLEN
            labelname = cutf.replace("_Energy_per_time.csv", "") + "_burnin-300"
            labelname = file2label(labelname)
            ax2.plot(rows[300:], burnin_acc_batch_step, label="{}".format(labelname))
            timesteps = np.where(burnin_acc_batch_step>=BASEACCBURNIN)
            if len(timesteps[0]) != 0 and ("Vth-1.0" not in f):
                ax2.axvline(rows[300+timesteps[0][0]], ymax=((BASEACCBURNIN-viewACCMIN)/(viewACCMAX-viewACCMIN)), linestyle="dotted", c='g')
        
        ax2.axhline(BASEACC, linestyle="dashed", color='#1f77b4')
        ax2.axhline(BASEACCBURNIN, linestyle="dotted", color='g')

    #ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xlabel("TimeStep")
    ax1.set_ylabel("Energy [J]")
    #ax1.set_title("(a)", y=-0.2)
    #ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xlabel("Energy [J]")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("(b)", y=-0.3)
    ax2.set_xlim(10, 98)
    ax2.set_ylim(viewACCMIN, viewACCMAX)
    fig.tight_layout()

    plt.rcParams["svg.fonttype"] = "none"

    savefile = os.path.join(args.csvDir, "Energy_Plot.svg")
    fig.savefig(savefile)

def file2label(labelname):
    labeln = labelname.replace("_", ", ")
    labeln = labeln.replace("-", "=")
    labeln = labeln.replace("alpha", "$\\alpha$")
    labeln = labeln.replace("beta", "$\\beta$")
    labeln = labeln.replace("burnin", "burn-in")
    outlabel = r"{}".format(labeln)
    return outlabel

if __name__ == "__main__":
    args = parser.parse_args()
    EachImage(args=args)
    EachTime(args=args)
    if args.Fire == 1:
        FireTime(args=args)
    Energy(args=args)
