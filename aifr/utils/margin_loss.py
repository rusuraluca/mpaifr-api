import torch
import torch.nn as nn
import torch.nn.functional as functional


class MarginLoss:
    @staticmethod
    def get_margin_loss(margin_loss_name, number_of_classes, embedding_size):
        margin_loss = {
            'cosface': CosFaceMarginLoss,
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

        margins = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), self.margin)
        logits -= margins

        logits *= self.scale

        return logits
