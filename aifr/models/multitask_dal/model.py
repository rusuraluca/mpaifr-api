import torch
import torch.nn as nn
import torch.nn.functional as functional

from ...utils.margin_loss import MarginLoss
from facenet_pytorch import InceptionResnetV1


class FRFM(nn.Module):
    def __init__(self, embedding_size=512):
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



class BCCA(nn.Module):
    def __init__(self, embedding_size=512):
        super(BCCA, self).__init__()
        self.id_predictor = nn.Linear(embedding_size, 1, bias=False)
        self.age_predictor = nn.Linear(embedding_size, 1, bias=False)
        self.gender_predictor = nn.Linear(embedding_size, 1, bias=False)

    def forward(self, id_features, age_features, gender_features):
        id_predictions = self.id_predictor(id_features)
        age_predictions = self.age_predictor(age_features)
        gender_predictions = self.gender_predictor(gender_features)

        id_mean = id_predictions.mean(dim=0)
        age_mean = age_predictions.mean(dim=0)
        gender_mean = gender_predictions.mean(dim=0)

        id_var = id_predictions.var(dim=0) + 1e-6
        age_var = age_predictions.var(dim=0) + 1e-6
        gender_var = gender_predictions.var(dim=0) + 1e-6

        id_age_corr = ((age_predictions - age_mean) * (id_predictions - id_mean)).mean(dim=0).pow(2) / (
                age_var * id_var)
        id_gender_corr = ((gender_predictions - gender_mean) * (id_predictions - id_mean)).mean(dim=0).pow(2) / (
                gender_var * id_var)
        age_gender_corr = ((age_predictions - age_mean) * (gender_predictions - gender_mean)).mean(dim=0).pow(2) / (
                age_var * gender_var)

        correlation_coefficient = (id_age_corr + id_gender_corr + age_gender_corr)/3

        return correlation_coefficient


class Multitask_DAL(nn.Module):
    def __init__(self, embedding_size=512, number_of_classes=1035, margin_loss_name='cosface', initializer=None):
        super(Multitask_DAL, self).__init__()

        self.backbone = InceptionResnetV1(pretrained='vggface2', dropout_prob=0.4)
        self.margin_loss = MarginLoss().get_margin_loss(margin_loss_name, number_of_classes, embedding_size)
        self.frfm = FRFM(embedding_size)
        self.bcca = BCCA(embedding_size)

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

