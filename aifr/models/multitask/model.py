import torch
import torch.nn as nn
import torch.nn.functional as functional

from ...utils.margin_loss import MarginLoss
from ...utils.metrics import compute_accuracy

from facenet_pytorch import InceptionResnetV1


class FRFM(nn.Module):
    def __init__(self, embedding_size):
        super(FRFM, self).__init__()
        self.age_transform = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(inplace=True),
        )

        self.gender_transform = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, embeddings):
        age_embeddings = self.age_transform(embeddings)
        gender_embeddings = self.gender_transform(embeddings)
        id_embeddings = embeddings - age_embeddings - gender_embeddings
        return id_embeddings, age_embeddings, gender_embeddings


class Multitask(nn.Module):
    def __init__(self, embedding_size=512, number_of_classes=1035, margin_loss_name='cosface', initializer=None):
        super(Multitask, self).__init__()

        self.backbone = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.4)
        self.margin_loss = MarginLoss().get_margin_loss(margin_loss_name, number_of_classes, embedding_size)
        self.frfm = FRFM(embedding_size)

        self.age_classifier = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 3),
        )

        self.gender_classifier = nn.Sequential(
            nn.Linear(embedding_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 2),
        )

        self.id_criterion = nn.CrossEntropyLoss()
        self.age_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None, age_groups=None, genders=None, return_embeddings=False):
        embeddings = self.backbone(inputs)
        id_embeddings, age_embeddings, gender_embeddings = self.frfm(embeddings)

        if return_embeddings:
            return functional.normalize(id_embeddings)

        id_logits = self.margin_loss(id_embeddings, labels)
        id_loss = self.id_criterion(id_logits, labels)
        id_accuracy = compute_accuracy(torch.max(id_logits, dim=1)[1], labels)

        age_logits = self.age_classifier(age_embeddings)
        age_loss = self.age_criterion(age_logits, age_groups)
        age_accuracy = compute_accuracy(torch.max(age_logits, dim=1)[1], age_groups)

        gender_logits = self.gender_classifier(gender_embeddings)
        gender_loss = self.gender_criterion(gender_logits, genders)
        gender_accuracy = compute_accuracy(torch.max(gender_logits, dim=1)[1], genders)

        return id_loss, id_accuracy, age_loss, age_accuracy, gender_loss, gender_accuracy
