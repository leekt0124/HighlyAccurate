import matplotlib.pyplot as plt
import os

OUTPUT_PATH = "./outputs"
DATE_PATH = "2023-04-11"
TIME_PATH = "02-32-55"

# _400 epochs with new LM setting
# DATE_PATH = "2023-04-10"
# TIME_PATH = "16-51-23"

# 1000 epochs with old LM setting
# DATE_PATH = "2023-04-09"
# TIME_PATH = "16-32-32"

file_path = os.path.join(OUTPUT_PATH, DATE_PATH, TIME_PATH)
loss_path = os.path.join(file_path, 'loss.txt')
plot_path = os.path.join(file_path, 'loss.png')

with open(loss_path, 'r') as f:
    lines = f.readlines()

lines = [float(l[:-1]) for l in lines]

plt.plot(lines)
plt.title("End-to-end loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig(plot_path)