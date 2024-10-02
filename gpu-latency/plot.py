#!/usr/bin/env python3

import os
import csv

import sys

sys.path.append("..")
from device_order import *


fig, ax = plt.subplots(figsize=(8, 4))
for filename in sorted(os.listdir("."), key=lambda f1: getOrderNumber(f1)):
    if not filename.endswith(".txt") or filename[:-4] not in order:
        continue
    with open(filename, newline="") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        sizes = []
        min = []
        max = []
        avg = []
        med = []
        for row in csvreader:
            if len(row) < 8 or row[0] == "clock:":
                continue
            sizes.append(float(row[2]))
            avg.append(float(row[4]))
            med.append(float(row[5]))
            min.append(float(row[6]))
            max.append(float(row[7]))

        print(filename, getOrderNumber(filename))

        ax.plot(
            sizes,
            med,
            # "-x",
            label=filename[:-4].upper(),
            color=getDeviceColor(filename),
            **lineStyle
        )

        plt.fill_between(
            sizes, min, max, alpha=0.4, color=getDeviceColor(filename), edgecolor=None
        )


ax.set_xlabel("chain data volume, kB")
ax.set_ylabel("latency, cycles")
ax.set_xscale("log", base=2)


# ax.axvline(16)
# ax.axvline(4*1024)

formatter = matplotlib.ticker.FuncFormatter(
    lambda x, pos: "{0:g} kB".format(x) if x < 1024 else "{0:g} MB".format(x // 1024)
)
ax.get_xaxis().set_major_formatter(formatter)
# ax.get_yaxis().set_major_formatter(formatter)

ax.set_xticks(
    [16, 128, 256, 4 * 1024, 6 * 1024, 20 * 1024, 40 * 1024, 128 * 1024, 512 * 1024]
)
ax.set_xlim([8, 800 * 1024])


ax.set_ylim([0, 800])

fig.autofmt_xdate()
ax.legend()
ax.set_ylim([0, ax.get_ylim()[1]])
fig.tight_layout()
fig.savefig("latency_plot.svg")

plt.show()
