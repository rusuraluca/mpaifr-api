import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from aifr.models.multitask_dal.model import Multitask_DAL


class ModelHandler:
    @staticmethod
    def get_model(model_name, dataset_name, margin_loss_name):
        model_type = {
            'multitask_dal': Multitask_DAL
        }

        dataset = {
            'big': 1035,
            'small': 500,
            'fgnet': 82,
        }

        if model_name.lower() in model_type:
            return model_type[model_name.lower()](
                embedding_size=512,
                number_of_classes=dataset[dataset_name],
                margin_loss_name=margin_loss_name,
            )
        else:
            raise ValueError("Unsupported model.")
