import os
import sys
from itertools import chain
import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import KFold

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(parent_dir)
from aifr.utils.metrics import compute_loss
from aifr.utils.image_loader import ImageLoader


class Multitask_DAL_Trainer:
    def __init__(self, model, config):
        self.device = torch.device('cpu')

        self.model = model.to(self.device)
        self.config = config

        self.learning_rate = self.config['learning_rate']
        self.batch_size = self.config['batch_size']
        self.lambdas =  self.config['lambdas']

        self.optimizer = optim.SGD(
            params=chain(
                self.model.margin_loss.parameters(),
                self.model.age_classifier.parameters(),
                self.model.gender_classifier.parameters(),
                self.model.frfm.parameters(),
                self.model.backbone.parameters(),
                self.model.bcca.parameters(),
            ),
            lr=self.learning_rate,
            momentum=0.9,
        )

        self.transforms_train = transforms.Compose([
            transforms.RandomResizedCrop((160, 160), scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transforms_test = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.CenterCrop(160),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.trainingDAL = False
        self.set_train_mode(False)

    @staticmethod
    def set_grads(mod, state):
        for param in mod.parameters():
            param.requires_grad = state

    def set_train_mode(self, state):
        self.trainingDAL = not state
        self.set_grads(self.model.margin_loss, True)
        self.set_grads(self.model.age_classifier, True)
        self.set_grads(self.model.gender_classifier, True)
        self.set_grads(self.model.frfm, state)
        self.set_grads(self.model.backbone, False)
        self.set_grads(self.model.bcca, not state)

    @staticmethod
    def flip_grads(mod):
        for param in mod.parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param.grad = -param.grad


    def load_data(self, training=True):
        path_key = 'train_root' if training else 'test_root'
        dataset_path = self.config.get(path_key)

        if dataset_path is None or not os.path.exists(dataset_path):
            print(f"Dataset path {dataset_path} does not exist.")
            return None

        if training:
            dataset = ImageLoader(
                root=dataset_path,
                transform=self.transforms_train)
        else:
            dataset = ImageLoader(
                root=dataset_path,
                transform=self.transforms_test)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def run_epoch(self, loader, training):
        if loader is None:
            print("Warning: DataLoader is None. Skipping this epoch.")
            return

        phase = 'Training' if training else 'Testing'
        self.model.train() if training else self.model.eval()

        for i, (images, labels, age_groups, genders) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            age_groups = age_groups.to(self.device)
            genders = genders.to(self.device)

            if training:
                if i % 70 < 60:
                    self.set_train_mode(False)
                else:
                    self.set_train_mode(True)

                id_loss, id_accuracy, age_loss, age_accuracy, gender_loss, gender_accuracy, cano_cor \
                    = self.model(inputs=images, labels=labels, age_groups=age_groups, genders=genders)

                total_loss = compute_loss(
                    id_loss,
                    age_loss,
                    gender_loss,
                    cano_cor,
                    lambdas=self.lambdas
                )

            else:
                with torch.no_grad():
                    id_loss, id_accuracy, age_loss, age_accuracy, gender_loss, gender_accuracy, cano_cor \
                        = self.model(inputs=images, labels=labels, age_groups=age_groups, genders=genders)

                    total_loss = compute_loss(
                        id_loss,
                        age_loss,
                        gender_loss,
                        cano_cor,
                        lambdas=self.lambdas
                    )

            if training:
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.trainingDAL:
                    self.flip_grads(self.model.bcca)
                self.optimizer.step()
                if self.trainingDAL:
                    self.flip_grads(self.model.bcca)

            metrics = {
                f"{phase}/total_loss": total_loss.item(),
                f"{phase}/id_loss": id_loss.item(),
                f"{phase}/id_accuracy": id_accuracy.item(),
                f"{phase}/age_loss": age_loss.item(),
                f"{phase}/age_accuracy": age_accuracy.item(),
                f"{phase}/gender_loss": gender_loss.item(),
                f"{phase}/gender_accuracy": gender_accuracy.item(),
                f"{phase}/cano_cor": cano_cor.item(),
                f"{phase}/progress": i / len(loader)
            }

            wandb.log(metrics)

    def save_model(self, epoch):
        model_path = os.path.join(self.config['save_model'], 'model_epoch_{}.pth'.format(epoch, epoch))
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def train(self, epochs):
        train_loader = self.load_data(training=True)
        test_loader = self.load_data(training=False)

        for epoch in range(epochs):
            print("Training")
            self.run_epoch(train_loader, training=True)

            if test_loader:
                print("Testing")
                self.run_epoch(test_loader, training=False)

            self.save_model(epoch)

    def run_epoch_leave_one_out(self, loader, training):
        if loader is None:
            print("Warning: DataLoader is None. Skipping this epoch.")
            return

        phase = 'Training' if training else 'Testing'
        self.model.train() if training else self.model.eval()

        metrics = {}
        accuracy = 0
        for i, (images, labels, age_groups, genders) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            age_groups = age_groups.to(self.device)
            genders = genders.to(self.device)

            if training:
                if i % 70 < 60:
                    self.set_train_mode(False)
                else:
                    self.set_train_mode(True)

                id_loss, id_accuracy, age_loss, age_accuracy, gender_loss, gender_accuracy, cano_cor \
                    = self.model(inputs=images, labels=labels, age_groups=age_groups, genders=genders)

                total_loss = compute_loss(
                    id_loss,
                    age_loss,
                    gender_loss,
                    cano_cor,
                    lambdas=self.lambdas
                )

            else:
                with torch.no_grad():
                    id_loss, id_accuracy, age_loss, age_accuracy, gender_loss, gender_accuracy, cano_cor \
                        = self.model(inputs=images, labels=labels, age_groups=age_groups, genders=genders)

                    total_loss = compute_loss(
                        id_loss,
                        age_loss,
                        gender_loss,
                        cano_cor,
                        lambdas=self.lambdas
                    )

            if training:
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.trainingDAL:
                    self.flip_grads(self.model.bcca)
                self.optimizer.step()
                if self.trainingDAL:
                    self.flip_grads(self.model.bcca)

            metrics = {
                f"{phase}/total_loss": total_loss.item(),
                f"{phase}/id_loss": id_loss.item(),
                f"{phase}/id_accuracy": id_accuracy.item(),
                f"{phase}/progress": i / len(loader)
            }
            wandb.log(metrics)
            if not training:
                accuracy = metrics['Testing/id_accuracy']

        return accuracy

    def train_leave_one_out(self, epochs, train_loader, test_loader):
        accuracy = 0
        for epoch in range(epochs):
            self.run_epoch_leave_one_out(train_loader, training=True)

            if test_loader:
                with torch.no_grad():
                    accuracy += self.run_epoch_leave_one_out(test_loader, training=False)
        return accuracy/epochs