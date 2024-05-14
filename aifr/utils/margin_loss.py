import torch
import torch.nn as nn
import torch.nn.functional as functional


class MarginLoss:
    @staticmethod
    def get_margin_loss(margin_loss_name, number_of_classes, embedding_size):
        margin_loss = {
            'cosface': CosFaceMarginLoss,
            'cosfaceV2': CosFaceV2MarginLoss,
            'arcface': ArcFaceMarginLoss
        }
        if margin_loss_name.lower() in margin_loss:
            return margin_loss[margin_loss_name.lower()](number_of_classes, embedding_size)
        else:
            raise ValueError("Unsupported loss head.")


class CosFaceMarginLoss(nn.Module):
    def __init__(self, number_of_classes, embedding_size=512, scale=32, margin=0.1):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weights = nn.Parameter(torch.Tensor(number_of_classes, embedding_size))
        nn.init.xavier_uniform_(self.weights)

    def forward(self, embeddings, labels):
        normalized_embeddings = functional.normalize(embeddings)
        normalized_weights = functional.normalize(self.weights)
        logits = functional.linear(normalized_embeddings, normalized_weights)

        if not self.training:
            return logits

        # Subtract margin from the logits at the correct class indices
        margins = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), self.margin)
        logits -= margins

        # Scale the logits
        logits *= self.scale

        return logits

class ArcFaceMarginLoss(nn.Module):
    def __init__(self, number_of_classes, embedding_size=512, scale=32.0, margin=0.1):
        super(ArcFaceMarginLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.embedding_weights = nn.Parameter(
            torch.Tensor(number_of_classes, embedding_size)
        )
        nn.init.xavier_uniform_(self.embedding_weights)

    def forward(self, embeddings, labels):
        logits = functional.linear(functional.normalize(embeddings), functional.normalize(self.embedding_weights))
        if not self.training:
            return logits
        return logits.scatter(
            1,
            labels.view(-1, 1),
            (logits.gather(1, labels.view(-1, 1)).acos() + self.margin).cos()
        ).mul(self.scale)

class CosFaceV2MarginLoss(nn.Module):
    def __init__(self, number_of_classes, embedding_size=512, scale=32.0, margin=0.1):
        super(CosFaceV2MarginLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.classifier = nn.Linear(embedding_size, number_of_classes, bias=False)
        nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, embeddings, labels):
        logits = self.classifier(embeddings.renorm(2, 0, 1e-5).mul(1e5))
        if not self.training:
            return logits
        return logits.scatter_add(
            1,
            labels.view(-1, 1),
            logits.new_full(labels.view(-1, 1).size(), -self.margin)
        ).mul(self.scale)
