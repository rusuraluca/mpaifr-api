import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as functional

from ...utils.margin_loss import MarginLoss
from ...utils.metrics import compute_accuracy

from facenet_pytorch import InceptionResnetV1


class Singletask(nn.Module):
    def __init__(self, embedding_size=512, number_of_classes=500, margin_loss_name='cosface', initializer=None):
        super(Singletask, self).__init__()

        self.backbone = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.4)
        self.margin_loss = MarginLoss().get_margin_loss(margin_loss_name, number_of_classes, embedding_size)
        self.id_criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None, return_embeddings=False):
        embeddings = self.backbone(inputs)

        if return_embeddings:
            return functional.normalize(embeddings)

        id_logits = self.margin_loss(embeddings, labels)
        id_loss = self.id_criterion(id_logits, labels)
        id_accuracy = compute_accuracy(torch.max(id_logits, dim=1)[1], labels)

        return id_loss, id_accuracy
