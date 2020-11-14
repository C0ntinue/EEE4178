# Evaluation

from train import ConvNet
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time
import seaborn as sn

from sklearn.metrics import roc_curve, auc
from scipy import interp
import itertools
from itertools import cycle


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

batch_size = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_test = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                                 ])

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform_test)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

cnn = ConvNet()
state=torch.load('model.pth')['net']
cnn.load_state_dict(state)

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = cnn(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

"""## Classification Report and Confusion matrix"""

def convert_image (image):
  return image.reshape((image.shape[0],-1))

# Reshape image input
# For each image with shape (32,32,3), reshape it to a single row of length 32*32*3
trainset_array = next(iter(torch.utils.data.DataLoader(trainset,batch_size=len(trainset))))[0].numpy()
testset_array = next(iter(torch.utils.data.DataLoader(testset,batch_size=len(testset))))[0].numpy()
train_X_array,test_X_array = convert_image(trainset_array), convert_image(testset_array)

# Add a column of 1's as the bias term
train_X_design = np.concatenate((np.ones((train_X_array.shape[0],1)), train_X_array), axis=1)
test_X_design = np.concatenate((np.ones((test_X_array.shape[0],1)), test_X_array), axis=1)
#train_X_design.shape

train_y_array = np.asarray(trainset.targets)
test_y_array = np.asarray(testset.targets)
#train_y_array.shape

from torch.autograd import Variable
def cnn_predict():
  y_score=[]  # predicted values for each instance
  y=[]
  y_predicted = []
  for data in testloader:
      images, labels = data
      images, labels = images.to(device), labels.to(device)
      outputs = cnn(Variable(images.to(device)))
      _, predicted = torch.max(outputs.data, 1)
      c = (predicted == labels.to(device)).squeeze().cpu().numpy()
      y_score.append(outputs.data.cpu().detach().numpy())
      y_predicted.append(predicted.cpu())
      y.append(labels.cpu().numpy())
  # concatenate every mini-batch result as an nparray
  y = np.concatenate(y)
  y_score = np.concatenate(y_score)
  y_predicted = np.concatenate(y_predicted)
  # one-hot encode the label y
  N = y.shape[0]
  C = 10
  y_hot = np.zeros((N,C))
  y_hot[np.arange(N), y] = 1
  # return y_predicted without one-hot encoding, y_label with one hot encoding, y_score which is the raw output in (10000,10)
  return y_predicted, y_hot, y_score



def plot_report(model, y_test, predictions):
    # classification report
    print(classification_report(y_true = y_test, y_pred = predictions, target_names = classes))
    # confusion matrix
    cm = confusion_matrix(y_true = y_test, y_pred = predictions, normalize = 'true')
    plt.figure(figsize=(10,8))
    sn.heatmap(cm)
    plt.title(model)
    plt.savefig(str(model)[0:7]+'_cm.jpg')
    plt.show()

def evaluation(X_test, y_test):
    print("--------------------------------------------------------------------------------------")
    print("CNN with learning rate = 0.001  dropout = 0.20")
    cnnPredict,_,_ = cnn_predict()
    plot_report(cnn, y_test, cnnPredict)

evaluation(test_X_design, test_y_array)

"""## ROC"""

from torch.autograd import Variable
y_score=[]
y=[]
for data in testloader:
    images, labels = data
    outputs = net(Variable(images.to(device)))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels.to(device)).squeeze().cpu().numpy()
    y_score.append(outputs.cpu().detach().numpy())
    y.append(labels.cpu().numpy())

y = np.concatenate(y)
y_score = np.concatenate(y_score)

N = y.shape[0]
C = 10
y_hot = np.zeros((N,C))
y_hot[np.arange(N), y] = 1

#Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_hot[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

#Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_hot.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

#Compute macro-average ROC curve and ROC area
#First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))

#Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(10):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

#Finally average it and compute AUC
mean_tpr /= 10

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

#Plot of a ROC curve for a specific class
plt.figure(figsize=(11, 9))
lw = 2
plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle='--', linewidth=4)
colors = sn.color_palette("Paired")
for i, color, j in zip(range(10), colors, classes):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(j, roc_auc[i]),linewidth=1.5)
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC&AUC')
plt.legend(loc="lower right")
plt.show()