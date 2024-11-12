import os
import datetime
import json
import logging
from ModelBuilder import ModelBuilder
from tensorflow.keras.models import save_model
from Trainer import Trainer
from DataLoader import DataLoader
from Evaluator import Evaluator
from HistoryManager import HistoryManager
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

c_handler = logging.StreamHandler()
f_handler = logging.FileHandler('timeseries_model.log')
c_handler.setLevel(logging.DEBUG)
f_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

logger.addHandler(c_handler)
logger.addHandler(f_handler)

class TimeSeriesModel:
    def __init__(self, file_path, base_dir="C:/AAA/", epochs=1, batch_size=16, block_configs=None):
        self.file_path = file_path
        self.base_dir = base_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.block_configs = block_configs

        os.makedirs(self.base_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.base_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.history_path = os.path.join(self.run_dir, 'training_history.json')
        self.model_save_path = os.path.join(self.run_dir, 'model.h5')
        self.loss_plot_path = os.path.join(self.run_dir, 'loss_plot.png')
        self.prediction_plot_path = os.path.join(self.run_dir, 'prediction_plot.png')
        self.error_distribution_plot_path = os.path.join(self.run_dir, 'error_distribution_plot.png')

        self.data_loader = DataLoader(file_path)
        self.model = None
        self.trainer = None
        self.evaluator = None

        logger.debug(f"Initialized TimeSeriesModel with run directory: {self.run_dir}")

    def save_plot(self, fig, path):
        if fig is not None:
            fig.savefig(path)
            plt.close(fig)
            logger.info(f"Plot saved successfully to {path}.")
        else:
            logger.error("Figure is None and cannot be saved.")

    def save_hyperparameters(self):
        hyperparams = {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'block_configs': self.block_configs
        }
        hyperparams_path = os.path.join(self.run_dir, 'hyperparameters.json')
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        logger.info(f"Hyperparameters saved to {hyperparams_path}")

    def save_model(self):
        keras_model_path = os.path.join(self.run_dir, 'model.keras')
        self.model.save(keras_model_path)
        logger.info(f"Model saved successfully to {keras_model_path}.")

        saved_model_dir = os.path.join(self.run_dir, 'saved_model')
        self.model.export(saved_model_dir)
        logger.info(f"Model exported successfully to {saved_model_dir}.")

    def plot_r2(self, r2_value):
        self.evaluator.plot_r2_bar(r2_value)

    def plot_metrics(self, history):
        mae_plot_path = os.path.join(self.run_dir, 'mae_plot.png')
        mse_plot_path = os.path.join(self.run_dir, 'mse_plot.png')
        rmse_plot_path = os.path.join(self.run_dir, 'rmse_plot.png')

        self.evaluator.plot_mae(history, mae_plot_path)
        self.evaluator.plot_mse(history, mse_plot_path)
        self.evaluator.plot_rmse(history, rmse_plot_path)

        logger.info("Metric plots have been saved successfully.")

    def save_model_per_epoch(self, epoch):
        keras_model_path = os.path.join(self.run_dir, f'model_epoch_{epoch:02d}.keras')
        self.model.save(keras_model_path)
        logger.info(f"Model saved to {keras_model_path} for epoch {epoch}.")

        saved_model_dir = os.path.join(self.run_dir, f'saved_model_epoch_{epoch:02d}')
        self.model.export(saved_model_dir)
        logger.info(f"Model exported to {saved_model_dir} for epoch {epoch}.")

    def run(self):
        try:



            X_train, X_test, y_train, y_test, scaler_y = self.data_loader.get_data()
            logger.info("Data loaded and preprocessed successfully.")

            self.save_hyperparameters()

            model_builder = ModelBuilder(
                time_steps=X_train.shape[1],
                num_features=X_train.shape[2],
                block_configs=self.block_configs 
            )
            self.model = model_builder.build_model()
            self.model.summary(print_fn=logger.debug)

            self.trainer = Trainer(
                model=self.model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=self.epochs,
                batch_size=self.batch_size,
                history_path=self.history_path,
                main_model_instance=self
            )

            history = self.trainer.train()
            logger.info("Model training completed.")
            logger.debug(f"Available keys in history: {history.history.keys()}")

            history_manager = HistoryManager(self.history_path)
            history_manager.save_history(history)
            logger.info(f"Training history saved to {self.history_path}.")

            self.evaluator = Evaluator(
                model=self.model,
                X_test=X_test,
                y_test=y_test,
                scaler_y=scaler_y,
                run_dir=self.run_dir,
                history_manager=history_manager
            )
            logger.info("Starting model evaluation...")

            self.plot_metrics(history)

            fig_loss = self.evaluator.plot_loss(history)
            self.save_plot(fig_loss, self.loss_plot_path)
            logger.info(f"Loss plot saved to {self.loss_plot_path}.")

            y_pred_rescaled, y_test_rescaled = self.evaluator.predict()
            logger.info("Predictions made and rescaled.")

            mae, mse, rmse, r2 = self.evaluator.calculate_metrics()
            self.evaluator.print_metrics(mae, mse, rmse, r2)
            logger.info("Evaluation metrics calculated and printed.")

            self.plot_r2(r2)
            logger.info("R² plot saved successfully.")

            fig_pred = self.evaluator.plot_predictions()
            self.save_plot(fig_pred, self.prediction_plot_path)
            logger.info(f"Prediction plot saved to {self.prediction_plot_path}.")

            fig_error = self.evaluator.plot_error_distribution()
            self.save_plot(fig_error, self.error_distribution_plot_path)
            logger.info(f"Error distribution plot saved to {self.error_distribution_plot_path}.")

            self.save_model()

        except Exception as e:
            logger.exception(f"An error occurred during the model pipeline execution: {e}")
