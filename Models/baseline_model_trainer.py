from Data import processor
from .baseline_models import Linear_Model, Tree_Ensemble_Model


#####################################################################
# This module trains the baseline models using standard conventions #
#####################################################################


class Trainer:

    # configurations
    config = processor.Baseline_CFG

    def __init__(self, stock, start, end, test_ratio, model):
        self.model = model
        self.data_params = {"stock": stock, "start": start, "end": end, "test_ratio": test_ratio}
        self.testing = None
        self.scaler = None

    def process_data(self):
        X_df, X_train, y_train, X_val, y_val, testing, scaler = processor.baseline_preprocessing(
            stock=self.data_params.get("stock"),
            start=self.data_params.get("start"),
            end=self.data_params.get("end"),
            test_ratio=self.data_params.get("test_ratio"),
        )
        self.testing = testing
        self.scaler = scaler
        return X_df, X_train, y_train, X_val, y_val, testing, scaler

    def train_model(self):
        _, X_train, y_train, X_val, y_val, _, _ = self.process_data()

        if hasattr(self.model, "train"):
            # If it's a Tree_Ensemble_Model
            if isinstance(self.model, Tree_Ensemble_Model):
                self.model.train(X_train, y_train, X_val, y_val)

            # If it's a Linear_Model
            elif isinstance(self.model, Linear_Model):
                self.model.train(X_train, y_train, X_val, y_val,
                                 learning_rate=0.01,
                                 epochs=50000,
                                 log_interval=1000)

            else:
                raise ValueError("Unsupported model class passed to Trainer.")
        else:
            raise ValueError("Model does not implement `train()` method.")
