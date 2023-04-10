import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv)<=1:
    print(f'Error! Format: python plot_ratio <y_limit>')
    sys.exit
else:
    ylim = float(sys.argv[1])

OUTPUT_PATH = "./outputs"
DATE_PATH = "2023-04-09"
TIME_PATH = "20-24-52"

file_path = os.path.join(OUTPUT_PATH, DATE_PATH, TIME_PATH)
loss_path = os.path.join(file_path, 'distance_ratio.txt')
plot_path = os.path.join(file_path, 'test_distance_ratio.png')

with open(loss_path, 'r') as f:
    lines = f.readlines()

lines = [float(l[:-1]) for l in lines]



plt.plot(lines)
# plt.plot(epochs, ave_ratios)
plt.xlabel('epoch')
plt.ylabel('pred_distance/init_distance')
plt.title('ratio of pred_dist/init_dis v.s. epoch')
plt.legend(['pred_shift_distance / init_shift_distance'])
plt.ylim([0, ylim])
# plt.axis('scaled')
plt.savefig(plot_path)