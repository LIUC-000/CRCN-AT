from torch.utils.data import TensorDataset, DataLoader
from model_complexcnn import *
import os
from get_dataset_label import *
import numpy as np
from sklearn.model_selection import train_test_split
os.environ['CUDA_VISIBLE_DEVICES']='3'
# os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import foolbox as fb
from foolbox.criteria import TargetedMisclassification

encoder = torch.load('./model_weight/Encoder_1_1_n_classes_10.pth')
classifier = torch.load('./model_weight/Classifier_1_1_n_classes_10.pth')


def Data_prepared(n_classes, k, RANDOM_SEED):
    X_train, X_val, X_test, value_Y_train, value_Y_val, value_Y_test = TrainDatasetKShotRound(n_classes, k, RANDOM_SEED)

    min_value = X_train.min()
    min_in_val = X_val.min()
    if min_in_val < min_value:
        min_value = min_in_val

    max_value = X_train.max()
    max_in_val = X_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    return max_value, min_value

def TestDataset_prepared(n_classes, k, RANDOM_SEED):
    X_train, X_val, X_test, value_Y_train, value_Y_val, value_Y_test = TrainDatasetKShotRound(n_classes, k, RANDOM_SEED)
    # X_test, value_Y_test = TestDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes, k, RANDOM_SEED)

    X_test = (X_test - min_value) / (max_value - min_value)

    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

    return X_test, value_Y_test

x, y = TestDataset_prepared(10, 10, 1)

test_dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)

class CVCNN(nn.Module):
    def __init__(self, encoder, classifier):
        super(CVCNN, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, data):
        features = self.encoder(data)
        output = F.log_softmax(self.classifier(features), dim=1)
        return output

model = CVCNN(encoder, classifier).cuda()

fmodel = fb.PyTorchModel(model,bounds=(-2,2))

attack = fb.attacks.LinfFastGradientAttack()

correct = 0
for data, target in test_dataloader:
    data = data.cuda()
    target = target.cuda()
    target = target.long()
    output = model(data)
    pred = output.argmax(dim=1, keepdim=True)
    # print(target[0:2], pred)
    correct += pred.eq(target.view_as(pred)).sum().item()

print(correct/len(test_dataloader.dataset))

model.eval()
correct = [0]*10
epsilons=[0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]
k = 0
for data, target in test_dataloader:
    data = data.cuda()
    target = target.cuda()
    target = target.long()
    # print(data[0:2,:,:].shape)
    # print(target)
    raw, clipped, is_adv = attack(fmodel, data, target, epsilons=epsilons)
    # print(raw.shape,clipped.shape)
    for i in range(10):
        output = model(clipped[i])
        pred = output.argmax(dim=1, keepdim=True)
        correct[i] += pred.eq(target.view_as(pred)).sum().item()
    # print(fb.utils.accuracy(fmodel, data, target))
    # print(fb.utils.accuracy(fmodel, raw, target))

print(correct)
print([i/len(test_dataloader.dataset) for i in correct])
