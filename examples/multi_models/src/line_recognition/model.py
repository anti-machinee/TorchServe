import torch.nn as nn


class ConvReLU(nn.Module):
    def __init__(self, in_size, out_size, kernel_size, stride, padding):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, stride, padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


class CNN(nn.Module):
    def __init__(self, in_size):
        super(CNN, self).__init__()
        self.conv_relu_1 = ConvReLU(in_size, 64, 3, 1, 1)
        self.max_pooling1 = nn.MaxPool2d(2)

        self.conv_relu_2 = ConvReLU(64, 128, 3, 1, 1)
        self.max_pooling2 = nn.MaxPool2d(2)

        self.conv_relu_3 = ConvReLU(128, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(256)

        self.conv_relu_4 = ConvReLU(256, 256, 3, 1, 1)
        self.max_pooling3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))

        self.conv_relu_5 = ConvReLU(256, 512, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(512)

        self.conv_relu_6 = ConvReLU(512, 512, 3, 1, 1)
        self.max_pooling4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1))

        self.conv_relu_7 = ConvReLU(512, 512, 2, 1, 0)
        self.bn3 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.max_pooling1(self.conv_relu_1(x))
        x = self.max_pooling2(self.conv_relu_2(x))
        x = self.bn1(self.conv_relu_3(x))
        x = self.max_pooling3(self.conv_relu_4(x))
        x = self.bn2(self.conv_relu_5(x))
        x = self.max_pooling4(self.conv_relu_6(x))
        x = self.bn3(self.conv_relu_7(x))
        return x


class BidirectionalLSTM(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=in_size, hidden_size=hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, out_size)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)
        output = output.view(T, b, -1)
        return output


class CRNN(nn.Module):
    def __init__(self, in_size, hidden_size, num_classes):
        super(CRNN, self).__init__()
        self.cnn = CNN(in_size)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_classes)
        )

    def forward(self, x):
        features = self.cnn(x)  # (bs, c, h, w)
        b, c, h, w = features.size()
        assert h == 1, "the height of features must be 1"
        features = features.squeeze(2)
        features = features.permute(2, 0, 1)  # (w, bs, c)
        sequence = self.rnn(features)
        return sequence
