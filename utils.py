import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pygame
from scipy.interpolate import interp1d


def signal_handler(sig, frame):
    print('Procedure terminated!')
    pygame.display.quit()
    pygame.quit()
    sys.exit(0)


def get_path():
    """
    Provides a prospective lateral-coordinate generator w.r.t possible
    longitudinal coordinates for the ego vehicle in Scenario 0.
    """
    waypoint_x_mark = np.array([200, 212.5, 225, 237.5, 250, 300])
    waypoint_y_mark = np.array([335, 336.5, 338, 337.5, 335, 334])
    pathgenerator = interp1d(waypoint_x_mark, waypoint_y_mark, kind='cubic')
    return pathgenerator


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, 1 / np.sqrt(y))
        m.bias.data.fill_(0)


class NET(nn.Module):
    def __init__(self):
        super(NET, self).__init__()
        self.relu  = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 6, 6)
        self.conv2 = nn.Conv2d(6, 16, 6)
        self.fc    = nn.Linear(16 * 7 * 16, 128)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.view(x.size(0), 16 * 16 * 7)
        x = self.fc(x)
        x = self.relu(x)
        return x


class RND(nn.Module):
    def __init__(self, use_cuda=True):
        super(RND, self).__init__()
        self.use_cuda  = use_cuda and torch.cuda.is_available()
        self.fix       = NET()
        self.estimator = NET()
        self.fix.apply(weights_init_normal)
        self.estimator.apply(weights_init_normal)
        self.criterion = nn.MSELoss()
        self.optim     = optim.Adam(self.estimator.parameters(), 0.0001)
        if self.use_cuda:
            self.fix.cuda()
            self.estimator.cuda()

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float).reshape(1, 45, 80)
        if self.use_cuda:
            state = state.cuda()
        target   = self.fix.forward(state)
        estimate = self.estimator.forward(state)
        loss     = self.criterion(estimate, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        error = loss.item()
        mu    = torch.mean(target)
        std   = torch.std(target)
        return error, mu.detach().cpu().numpy(), std.detach().cpu().numpy()

    def get_reward_i(self, state):
        error, mu, std = self.forward(state)
        alpha = 1 + (error - mu) / std
        return min(max(alpha, 1), 2)