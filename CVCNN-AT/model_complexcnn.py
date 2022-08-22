import torch
from  torch import nn
import torch.nn.functional as F
from complexcnn import ComplexConv, ComplexConv_trans
# from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = ComplexConv(in_channels=1,out_channels=64,kernel_size=4,stride=2)
        self.batchnorm1 = nn.BatchNorm1d(num_features=128)
        self.conv2 = ComplexConv(in_channels=64,out_channels=64,kernel_size=4,stride=2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128)
        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm3 = nn.BatchNorm1d(num_features=128)
        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm4 = nn.BatchNorm1d(num_features=128)
        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm5 = nn.BatchNorm1d(num_features=128)
        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm6 = nn.BatchNorm1d(num_features=128)
        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm7 = nn.BatchNorm1d(num_features=128)
        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm8 = nn.BatchNorm1d(num_features=128)
        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=4,stride=2)
        self.batchnorm9 = nn.BatchNorm1d(num_features=128)
        self.flatten = nn.Flatten()
        self.linear1 = nn.LazyLinear(1024)

    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)

        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm8(x)

        x = self.conv9(x)
        x = F.relu(x)
        x = self.batchnorm9(x)

        x = self.flatten(x)
        x = self.linear1(x)
        embedding = F.relu(x)
        return embedding

# encoder = Encoder()
# summary(encoder,(2,6000))


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.linear1 = nn.LazyLinear(1152)
        self.batchnorm1 = nn.BatchNorm1d(num_features=128)

        self.conv1 = ComplexConv_trans(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=128)

        self.conv2 = ComplexConv_trans(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.batchnorm3 = nn.BatchNorm1d(num_features=128)

        self.conv3 = ComplexConv_trans(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.batchnorm4 = nn.BatchNorm1d(num_features=128)

        self.conv4 = ComplexConv_trans(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.batchnorm5 = nn.BatchNorm1d(num_features=128)

        self.conv5 = ComplexConv_trans(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.batchnorm6 = nn.BatchNorm1d(num_features=128)

        self.conv6 = ComplexConv_trans(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.batchnorm7 = nn.BatchNorm1d(num_features=128)

        self.conv7 = ComplexConv_trans(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.batchnorm8 = nn.BatchNorm1d(num_features=128)

        self.conv8 = ComplexConv_trans(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.batchnorm9 = nn.BatchNorm1d(num_features=128)

        self.conv9 = ComplexConv_trans(in_channels=64, out_channels=1, kernel_size=4, stride=2)



    def forward(self,x):
        x = self.linear1(x)
        x = x.view(-1, 128, 9)
        x = self.batchnorm1(x)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm2(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm3(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm4(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm5(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm6(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm7(x)

        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm8(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm9(x)

        x = self.conv9(x)
        # x = F.relu(x)

        x = F.sigmoid(x)

        return x

# decoder = Decoder()
# summary(decoder,(1024,))

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(1024,10)

    def forward(self,x):
        x = self.linear(x)
        return x

if __name__ == "__main__":
    coder = Encoder()
    decoder = Decoder()
    classifier = Classifier()

    input = torch.randn((10,2,6000))

    features = coder(input)
    re_input = decoder(features)
    output = F.log_softmax(classifier(features))

    print(features.shape)
    print(re_input.shape)
    print(output.shape)



