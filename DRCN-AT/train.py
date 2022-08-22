from AT import AT_X
from get_dataset_label import *
import argparse
from torch.utils.data import TensorDataset, DataLoader
from model_complexcnn import *
import os
import random

sample_seed = 1
RANDOM_SEED = 1 # any random number
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
set_seed(RANDOM_SEED)

os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser(description='PyTorch Complex_test Training')
parser.add_argument('--lr_encoder', type=float, default=0.001, metavar='LR:0.1 SVHN:0.01',
                    help='learning rate')
parser.add_argument('--lr_decoder', type=float, default=0.001, metavar='LR:0.1 SVHN:0.01',
                    help='learning rate')
parser.add_argument('--lr_classifier', type=float, default=0.001, metavar='LR:0.1 SVHN:0.01',
                    help='learning rate')
args = parser.parse_args(args=[])

def train(encoder, decoder, classifier, loss_nll, loss_mse, train_dataloader, optimizer_encoder, optimizer_decoder, optimizer_classifier, epoch, device):
    encoder.train()  # 启动训练, 允许更新模型参数
    decoder.train()
    classifier.train()
    correct = 0
    nll_loss = 0
    mse_loss = 0
    X_adv_gen = AT_X(encoder, classifier)
    for data, target in train_dataloader:
        target = target.long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)

        optimizer_encoder.zero_grad()  # 清空优化器中梯度信息
        optimizer_decoder.zero_grad()
        optimizer_classifier.zero_grad()

        lamda_re = 1.875
        lamda_ce = 0.125

        data_adv = X_adv_gen(data, target)

        # 分类损失反向生成encoder和classifier的梯度
        features = encoder(data)
        output = F.log_softmax(classifier(features), dim=1)
        features_adv = encoder(data_adv)
        output_adv = F.log_softmax(classifier(features_adv), dim=1)
        nll_loss_batch = lamda_ce * loss_nll(output, target)) + lamda_ce * loss_nll(output_adv, target)
        nll_loss_batch.backward()

        # 重构损失反向生成encoder和decoder的梯度
        features = encoder(data)
        re_input = decoder(features)
        features_adv = encoder(data_adv)
        re_input_adv = decoder(features_adv)
        mse_loss_batch = lamda_re * loss_mse(re_input, data) + lamda_re * loss_mse(re_input_adv, data_adv)
        mse_loss_batch.backward()

        optimizer_encoder.step()
        optimizer_classifier.step()
        optimizer_decoder.step()

        nll_loss += nll_loss_batch.item()
        mse_loss += mse_loss_batch.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()  # 求pred和target中对应位置元素相等的个数

    nll_loss /= len(train_dataloader.dataset)
    mse_loss /= len(train_dataloader.dataset)

    print('Train Epoch: {} \tClass_Loss: {:.6f}, Recon_loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        nll_loss,
        mse_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )

def evaluate(encoder, classifier, loss_nll, val_dataloader, device):
    encoder.eval()  # 启动验证，不允许更新模型参数
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = classifier(encoder(data))
            output = F.log_softmax(output, dim=1)
            test_loss += loss_nll(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_dataloader.dataset)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(val_dataloader.dataset),
            100.0 * correct / len(val_dataloader.dataset),
        )
    )

    return test_loss

def test(encoder, classifier, test_dataloader, device):
    encoder.eval()  # 启动验证，不允许更新模型参数
    classifier.eval()
    test_loss = 0
    correct = 0
    loss = nn.NLLLoss()
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
                loss = loss.to(device)
            output = classifier(encoder(data))
            output = F.log_softmax(output, dim=1)
            test_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_dataloader.dataset)
    fmt = '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    # if epoch >= 50:
    #     lr = args.lr * 0.1
    # if epoch >= 200:
    #     lr = args.lr * 0.01
    if epoch >= 100:
        lr = args.lr * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_and_test(encoder, decoder, classifier, loss_nll, loss_mse, train_dataloader, val_dataloader, optim_encoder, optim_decoder, optim_classifier, epochs, save_path_encoder, save_path_classifier, device):
    current_min_val_loss = 100000000
    for epoch in range(1, epochs + 1):
        # adjust_learning_rate(optimizer, epoch)
        train(encoder, decoder, classifier, loss_nll, loss_mse, train_dataloader, optim_encoder, optim_decoder, optim_classifier, epoch, device)
        val_loss = evaluate(encoder, classifier, loss_nll, val_dataloader, device)
        if val_loss < current_min_val_loss:
            print("The validation loss is improved from {} to {}, new model weight is saved.".format(
                current_min_val_loss, val_loss))
            current_min_val_loss = val_loss
            torch.save(encoder, save_path_encoder)
            torch.save(classifier, save_path_classifier)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

def Data_prepared(n_classes, k, sample_seed):
    X_train, X_val, X_test, value_Y_train, value_Y_val, value_Y_test = TrainDatasetKShotRound(n_classes, k, sample_seed)

    min_value = X_train.min()
    min_in_val = X_val.min()
    if min_in_val < min_value:
        min_value = min_in_val

    max_value = X_train.max()
    max_in_val = X_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    return max_value, min_value

def TrainDataset_prepared(n_classes, k, sample_seed):
    X_train, X_val, X_test, value_Y_train, value_Y_val, value_Y_test = TrainDatasetKShotRound(n_classes, k, sample_seed)
    # X_test, value_Y_test = TestDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes, k, sample_seed)

    X_train = (X_train - min_value) / (max_value - min_value)
    X_val = (X_val - min_value) / (max_value - min_value)
    X_test = (X_test - min_value) / (max_value - min_value)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[2], X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[1])

    return X_train, X_val, X_test, value_Y_train, value_Y_val, value_Y_test

def run(train_dataloader, val_dataloader, epochs, save_path_encoder, save_path_classifier, device):
    encoder = Encoder()
    decoder = Decoder()
    classifier = Classifier()

    if torch.cuda.is_available():
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        classifier = classifier.to(device)

    loss_nll = nn.NLLLoss()
    loss_mse = nn.MSELoss()
    if torch.cuda.is_available():
        loss_nll = loss_nll.to(device)
        loss_mse = loss_mse.to(device)

    optim_encoder = torch.optim.Adam(encoder.parameters(), lr=args.lr_encoder, weight_decay=0.0001)
    optim_decoder = torch.optim.Adam(decoder.parameters(), lr=args.lr_decoder, weight_decay=0.0001)
    optim_classifier = torch.optim.Adam(classifier.parameters(), lr=args.lr_classifier, weight_decay=0.0001)

    train_and_test(encoder, decoder, classifier, loss_nll, loss_mse, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                   optim_encoder=optim_encoder, optim_decoder=optim_decoder, optim_classifier=optim_classifier, epochs=epochs, save_path_encoder=save_path_encoder, save_path_classifier=save_path_classifier, device=device)

class Config:
    def __init__(
        self,
        batch_size: int = 10,
        val_batch_size: int = 16,
        test_batch_size: int = 16,
        epochs: int = 300,
        log_interval: int = 10,
        n_classes: int = 10,
        k_shot: int = 10,
        save_path_encoder: str = 'model_weight/Encoder_1_1_n_classes_10.pth',  # train seed, sample seed
        save_path_classifier: str = 'model_weight/Classifier_1_1_n_classes_10.pth',
        ):
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.log_interval = log_interval
        self.n_classes = n_classes
        self.k_shot = k_shot
        self.save_path_encoder = save_path_encoder
        self.save_path_classifier = save_path_classifier


def main():
    conf = Config()
    X_train, X_val, X_test, value_Y_train, value_Y_val, Y_test = TrainDataset_prepared(conf.n_classes, conf.k_shot,
                                                                                       sample_seed)
    device = torch.device("cuda:0")

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(value_Y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(value_Y_val))
    val_dataloader = DataLoader(val_dataset, batch_size=conf.val_batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=conf.test_batch_size, shuffle=True)

    # train
    run(train_dataloader, val_dataloader, epochs=conf.epochs, save_path_encoder=conf.save_path_encoder,
        save_path_classifier=conf.save_path_classifier, device=device)


    print("Test_result:")
    encoder = torch.load(conf.save_path_encoder)
    classifier = torch.load(conf.save_path_classifier)
    encoder = encoder.to(device)
    classifier = classifier.to(device)
    test(encoder, classifier, test_dataloader, device)

if __name__ == '__main__':
   main()