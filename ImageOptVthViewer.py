import argparse
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description='csv analystic')
parser.add_argument('--csvDir', default=None)


def getAccIndex(rows):
    r_data = np.array(rows[1:], dtype=np.float32)
    acc_index = r_data[:, -1] == 1
    y = r_data[acc_index, 0]
    x = r_data[acc_index, 1]

    return x, y


def getAccPerClass(rows):
    data_num = len(rows)
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


def main(args):
    all_files = os.listdir(args.csvDir)
    csv_files = [f for f in all_files if ".csv" in f]
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
        print("File {}, Acc {}".format(f, acc[0]))
    
    PerImage_x = np.concatenate(PerImage_x_list)
    PerImage_y = np.concatenate(PerImage_y_list)
    
    # Accurate plot per image
    ax1.scatter(PerImage_x, PerImage_y, s=0.5)
    ax2.set_xticks(class_idx)
    ax2.legend()

    savefile = os.path.join(args.csvDir, "Acc_Analay_Plot.png")
    fig.savefig(savefile, dpi=150)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args=args)
