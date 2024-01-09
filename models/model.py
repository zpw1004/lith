import torch.nn as nn
import torch
from senet import SEBlock
import torch.nn.functional as F

from util import parse_arguments


class MultiScaleNetwork(nn.Module):
    def __init__(self,num_classes,features,dropout=0):
        super(MultiScaleNetwork, self).__init__()
        self.conv = nn.ModuleList([
            nn.Conv1d(1, 16, 1, padding=0),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(3, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Conv1d(5, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU()
        ])
        self.weights = nn.Parameter(torch.randn(3), requires_grad=True)
        self.seblock = SEBlock(48, reduction=16)
        self.feature_refinement_module = nn.Sequential(
            nn.Conv1d(48, 16, 1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(16, 8, 3, padding=1),
            nn.Conv1d(8, 16, 3, padding=1),
        )
        self.output_layer = nn.Sequential(
            nn.Dropout(dropout),  # Add dropout here
            nn.Flatten(),
            nn.BatchNorm1d(16 * features),
            nn.LeakyReLU(),
            nn.Linear(16 * features, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128,num_classes)
        )

    def forward(self, x):
        convs = []
        dynamic_weights = F.softmax(self.weights, dim=0)
        for i in range(0, len(self.conv), 3):
            conv_layer = self.conv[i]
            batch_norm = self.conv[i + 1]
            activation = self.conv[i + 2]
            conv = activation(batch_norm(conv_layer(x[i // 3])))
            conv *= dynamic_weights[i // 3]
            convs.append(conv)
        x = torch.cat(convs, dim=1)
        x = self.seblock(x)
        x = self.feature_refinement_module(x)
        x = self.output_layer(x)
        return x
args = parse_arguments()
device = torch.device("cuda:0" if args.cuda else "cpu")
class multi_FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha_t=None):
        super(multi_FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha_t = torch.tensor(alpha_t).to(device) if alpha_t else None
        self.gamma = nn.Parameter(torch.zeros(num_classes, device=device), requires_grad=True)

    def forward(self, outputs, targets):
        if self.alpha_t is None:
            alpha_t = torch.ones(self.num_classes).to(outputs.device)
        else:
            alpha_t = self.alpha_t.to(outputs.device)

        ce_loss = nn.functional.cross_entropy(outputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma[targets] * ce_loss * alpha_t[targets]).mean()
        return focal_loss

