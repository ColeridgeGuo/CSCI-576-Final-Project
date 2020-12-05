import os
import matplotlib.pyplot as plt
import json

root_path = "/Users/yingxuanguo/Documents/USC/CSCI-576/Final Project/descriptor"
test_path = os.path.join(root_path, "test_descriptor.json")
train_path = os.path.join(root_path, "train_descriptor.json")

with open(train_path, 'r') as train_file, open(test_path, 'r') as test_file:
    train_data = json.load(train_file)
    test_data = json.load(test_file)

def plot_descriptor(y, test_name, train_name, ylim=(0.5, 1.0), save=False):
    x = range(0, 480)
    plt.figure(figsize=(8.88, 3))
    plt.plot(x, y, color='red', alpha=0.1)
    plt.fill_between(x, 0, y, color='red', alpha=.4)
    plt.ylim(ylim)
    plt.axis('off')
    if save:
        plt.savefig(os.path.join(test_name, f"{train_name}_descriptor.png"))
    plt.show()


test = {vid: test_data[cat][vid] for cat in test_data for vid in test_data[cat]}
train = {vid: train_data[cat][vid] for cat in train_data for vid in train_data[cat]}

def calc_descriptor(test_name, train_name) -> list:
    descriptor_score = []
    for i in range(0, 480):
        test_val, train_val = test[test_name][i], train[train_name][i]
        descriptor_score.append(min(test_val, train_val) / max(test_val, train_val))
    return descriptor_score


with open(os.path.join(root_path, 'top_5.json'), 'r') as f:
    top_5 = json.load(f)

for test_vid in top_5:
    for train_vid in top_5[test_vid]:
        des = calc_descriptor(test_vid, train_vid)
        plot_descriptor(des, test_vid, train_vid, ylim=(min(des), 1), save=True)

