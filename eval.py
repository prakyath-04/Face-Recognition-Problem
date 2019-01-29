import os,time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


train_path = "./sample_train_dataset.txt"
test_path = "./sample_test_dataset.txt"
gt_path = "./output_of_sample_test_dataset.txt"
pred_path = "./pred.txt"

# run classifier and store output in pred_path
script_path = "./linear_classifier.py"
# script_path = "./naive_bayes.py"
os.system("python2.7 {} {} {} > {}".format(script_path, train_path, test_path, pred_path))

"""
We will compute precision, accuracy and F1 score.
For that we require that all the output labels are valid.

For example, if train file contains only three labels : alice, bob and abc
and if your output is some string other than these three labels then it will be considered invalid and you will get zero marks.
"""
# collect all labels of train file
# ---------------------------------
with open(train_path, "r") as f:
    lines = f.readlines()

label_set = set()
for l in lines:
    img_path, label = l.strip().split()
    label_set.add(label)
# ---------------------------------


# validate the predictions
# ---------------------------------
# time.sleep(10)
with open(pred_path, "r") as f:
    pred_labels = f.readlines()
    pred_labels = [l.strip() for l in pred_labels]
# time.sleep(10)
with open(gt_path, "r") as f:
    gt_labels = f.readlines()
    gt_labels = [l.strip() for l in gt_labels]
# time.sleep(10)
for gtl in gt_labels:
    assert(gtl in label_set)
# time.sleep(10)
for predl in pred_labels:
    if predl not in label_set:
        print("Invalid output")
        print("All scores zero")
        exit(0)
# ---------------------------------


# evaluate the predictions
# ---------------------------------
mp = {l : idx for idx, l in enumerate(label_set)}
gt_labels = [mp[l] for l in gt_labels]
pred_labels = [mp[l] for l in pred_labels]

print(f1_score(gt_labels, pred_labels, average="macro"))
print(precision_score(gt_labels, pred_labels, average="macro"))
print(recall_score(gt_labels, pred_labels, average="macro"))
# ---------------------------------
