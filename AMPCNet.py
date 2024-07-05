import torch
import torch.nn as nn
    
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class PSABlock(nn.Module):
    def __init__(self, in_channels):
        super(PSABlock, self).__init__()
        self.group_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=4) 
        self.conv_list = nn.ModuleList([
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, padding=1),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=5, padding=2),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=7, padding=3),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=9, padding=4)
        ])
        self.se_blocks = nn.ModuleList([SEBlock(in_channels // 4) for _ in range(4)])

    def forward(self, x):
        x = self.group_conv(x)
        x_split = torch.split(x, x.size(1) // 4, dim=1)

        outputs = []
        for x_i, conv_i, se_i in zip(x_split, self.conv_list, self.se_blocks):
            att = se_i(conv_i(x_i))
            outputs.append(x_i * att)

        return torch.cat(outputs, dim=1)
    
class TemporalConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TemporalConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), stride=(2, 1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class TimeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TimeConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 10))  
        self.unit1 = TemporalConvUnit(out_channels, out_channels)
        self.unit2 = TemporalConvUnit(out_channels, out_channels)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.se(x)
        return x

class FreqConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FreqConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(10, 1), stride=(2, 15))  
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.se(x)
        return x

class TFConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TFConvUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(2, 15))
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(2, 15))
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.25)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class TFConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TFConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), padding=(2, 2))  
        self.unit1 = TFConvUnit(out_channels, out_channels)
        self.unit2 = TFConvUnit(out_channels, out_channels)
        self.psa = PSABlock(out_channels) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.psa(x)
        return x

class AMPCNet(nn.Module):
    def __init__(self, num_classes): 
        super().__init__()

        time_conv_channels = 16 
        freq_conv_channels = 16
        tf_conv_channels = 16
        input_channel = 2002

        self.time_conv = TimeConvBlock(2002, time_conv_channels)
        self.freq_conv = FreqConvBlock(8, freq_conv_channels)
        self.tf_conv = TFConvBlock(8, tf_conv_channels)

        flatten_size = flatten_size = time_conv_channels * 2 + freq_conv_channels * 2  + tf_conv_channels * 2

        self.classifier = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(flatten_size, 128),  
            nn.ReLU(),
            nn.Linear(128, num_classes) 
        )

    # def forward(self, x):
    #     x_time = self.time_conv(x)
    #     x_freq = self.freq_conv(x)
    #     x_tf = self.tf_conv(x)

    #     att_time = torch.matmul(x_time, x_time.transpose(-2, -1))
    #     att_freq = torch.matmul(x_freq, x_freq.transpose(-2, -1))
    #     att_tf = torch.matmul(x_tf, x_tf.transpose(-2, -1))

    #     att = torch.softmax(att_time + att_freq + att_tf, dim=-1)

    #     x_tf = x_tf * att
    #     x_time = x_time * att
    #     x_freq = x_freq * att 


    #     x_time = x_time.flatten(start_dim=1)
    #     x_freq = x_freq.flatten(start_dim=1) 
    #     x_tf = x_tf.flatten(start_dim=1) 

    #     concat_features = torch.cat([x_time, x_freq, x_tf], dim=1)

    #     out = self.fc1(concat_features)
    #     out = self.fc2(out)
    #     out = nn.Softmax(dim=1)(out)
    #     return out
    def forward(self, x):
        print("Input to time_conv:", x.shape)        
        x_time = self.time_conv(x)
        print("Output of time_conv:", x_time.shape)  
        x_freq = self.freq_conv(x)
        x_tf = self.tf_conv(x)

        att_time = torch.matmul(x_time, x_time.transpose(-2, -1))
        att_freq = torch.matmul(x_freq, x_freq.transpose(-2, -1))
        att_tf = torch.matmul(x_tf, x_tf.transpose(-2, -1))

        att = torch.softmax(att_time + att_freq + att_tf, dim=-1)

        x_tf = x_tf * att
        x_time = x_time * att
        x_freq = x_freq * att 


        x_time = x_time.flatten(start_dim=1)
        x_freq = x_freq.flatten(start_dim=1) 
        x_tf = x_tf.flatten(start_dim=1) 

        concat_features = torch.cat([x_time, x_freq, x_tf], dim=1)

        out = self.classifier(concat_features)
        # out = self.fc2(out)
        # out = nn.Softmax(dim=1)(out)
        return out