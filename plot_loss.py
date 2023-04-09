import matplotlib.pyplot as plt
import os

OUTPUT_PATH = "./outputs"
DATE_PATH = "2023-04-08"
TIME_PATH = "23-55-32"

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