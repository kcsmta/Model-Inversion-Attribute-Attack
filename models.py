import torch
import torch.nn as nn


class Covid19_MainTaskModel(nn.Module):
    def __init__(self):
        super(Covid19_MainTaskModel, self).__init__()
        self.fc1 = nn.Linear(8, 6)
        self.fc2 = nn.Linear(6, 2)
 
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        # out = torch.sigmoid(out)
        out = torch.softmax(out, dim=1)
        return out


class Adults_MainTaskModel(nn.Module):
    def __init__(self):
        super(Adults_MainTaskModel, self).__init__()
        self.fc1 = nn.Linear(14, 8)
        self.fc2 = nn.Linear(8, 2)
 
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        # out = torch.sigmoid(out)
        out = torch.softmax(out, dim=1)
        return out

class Fivethirtyeight_MainTaskModel(nn.Module):
    def __init__(self):
        super(Fivethirtyeight_MainTaskModel, self).__init__()
        self.fc1 = nn.Linear(12, 8)
        self.fc2 = nn.Linear(8, 5)
 
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        # out = torch.sigmoid(out)
        out = torch.softmax(out, dim=1)
        return out


class GSS_MainTaskModel(nn.Module):
    def __init__(self):
        super(GSS_MainTaskModel, self).__init__()
        self.fc1 = nn.Linear(11, 7)
        self.fc2 = nn.Linear(7, 3)
        # self.fc3 = nn.Linear(5, 3)
 
    def forward(self, x):
        out = self.fc1(x)
        # out = torch.sigmoid(out)
        out = self.fc2(out)
        # out = self.fc3(out)
        # out = torch.sigmoid(out)
        out = torch.softmax(out, dim=1)
        return out