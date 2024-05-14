import torch

def compute_accuracy(predictions, labels):
    with torch.no_grad():
        return (predictions.squeeze() == labels.squeeze()).float().mean()


def compute_loss(id_loss, age_loss=0, gender_loss=0, cano_corr=0, lambdas=(1, 1, 1)):
    return id_loss + lambdas[0] * age_loss + lambdas[1] * gender_loss + lambdas[2] * cano_corr
