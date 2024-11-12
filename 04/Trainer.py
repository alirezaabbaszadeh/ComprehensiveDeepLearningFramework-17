import tensorflow as tf
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from Evaluator import Evaluator
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('PIL.PngImagePlugin').disabled = True
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

class Trainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, epochs=1, batch_size=16, history_path=None, main_model_instance=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = None
        self.history_path = history_path
        self.main_model_instance = main_model_instance
        logging.debug("Trainer initialized with model: %s", model)

    def train(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            self.SaveHistoryCallback(self.history_path),
            self.SaveModelPerEpochCallback(self.main_model_instance),
            self.SavePlotsPerEpochCallback(self.main_model_instance, self.X_val, self.y_val)
            # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

        ]
        
        logging.info("Starting model training...")
    
        try:
            self.history = self.model.fit(
                self.X_train, self.y_train,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(self.X_val, self.y_val),
                verbose=1,
                callbacks=callbacks
            )
            logging.info("Model training completed.")
    
            if self.history is None or not hasattr(self.history, 'history'):
                logging.warning("No history generated during training or 'history' attribute is missing.")
            else:
                logging.debug("Available keys in history: %s", self.history.history.keys())
            
            return self.history
    
        except Exception as e:
            logging.error("An error occurred during training: %s", e)
            raise e

    class SaveHistoryCallback(tf.keras.callbacks.Callback):
        def __init__(self, history_path):
            super().__init__()
            self.history_path = history_path
            logging.debug("SaveHistoryCallback initialized with path: %s", history_path)

        def on_train_end(self, logs=None):
            history_dict = self.model.history.history
            with open(self.history_path, 'w') as f:
                json.dump(history_dict, f)
            logging.info("Training history saved to %s", self.history_path)

    class SaveModelPerEpochCallback(tf.keras.callbacks.Callback):
        def __init__(self, main_model_instance):
            super().__init__()
            self.main_model_instance = main_model_instance
            logging.debug("SaveModelPerEpochCallback initialized.")

        def on_epoch_end(self, epoch, logs=None):
            if self.main_model_instance is None:
                logging.error("Main model instance is None.")
                return
            self.main_model_instance.save_model_per_epoch(epoch + 1)
            logging.debug("Model saved at epoch %d", epoch + 1)

    class SavePlotsPerEpochCallback(tf.keras.callbacks.Callback):
        def __init__(self, main_model_instance, X_val, y_val):
            super().__init__()
            self.main_model_instance = main_model_instance
            self.X_val = X_val
            self.y_val = y_val
            self.train_losses = []
            self.val_losses = []
            logging.debug("SavePlotsPerEpochCallback initialized.")

        def on_epoch_end(self, epoch, logs=None):
            try:
                if logs is None:
                    logs = {}
                
                if self.X_val is None or self.y_val is None:
                    logging.error("Validation data (X_val or y_val) is None.")
                    return

                if self.main_model_instance is None:
                    logging.error("Main model instance is None.")
                    return

                if self.main_model_instance.data_loader is None or self.main_model_instance.data_loader.scaler_y is None:
                    logging.error("Data loader or scaler_y is None.")
                    return

                logging.debug("X_val shape: %s", self.X_val.shape)
                logging.debug("y_val shape: %s", self.y_val.shape)

                train_loss = logs.get('loss')
                val_loss = logs.get('val_loss')
                
                if train_loss is None or val_loss is None:
                    logging.warning("Epoch %d: 'loss' or 'val_loss' is missing.", epoch + 1)
                    return

                self.train_losses.append(train_loss)
                self.val_losses.append(val_loss)

                epoch_dir = os.path.join(self.main_model_instance.run_dir, f'epoch_{epoch + 1:02d}')
                os.makedirs(epoch_dir, exist_ok=True)

                # Skip plotting on the first epoch
                if epoch == 0:
                    logging.warning("Skipping plot creation on first epoch as predictions may not be available yet.")
                    return

                try:
                    logging.debug("Predicting on validation data...")
                    y_pred = self.model.predict(self.X_val, verbose=0)
                    if y_pred is None:
                        logging.error("Predictions are None.")
                        return

                    y_pred_rescaled = self.main_model_instance.data_loader.scaler_y.inverse_transform(y_pred)
                    y_val_rescaled = self.main_model_instance.data_loader.scaler_y.inverse_transform(self.y_val)
                    logging.debug("y_pred_rescaled shape: %s", y_pred_rescaled.shape)
                    logging.debug("y_val_rescaled shape: %s", y_val_rescaled.shape)
                except Exception as e:
                    logging.error("Error during prediction or scaling: %s", e)
                    return

                try:
                    mae = mean_absolute_error(y_val_rescaled, y_pred_rescaled)
                    mse = mean_squared_error(y_val_rescaled, y_pred_rescaled)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_val_rescaled, y_pred_rescaled)
                    logging.info("Metrics at epoch %d - MAE: %f, MSE: %f, RMSE: %f, R2: %f", epoch + 1, mae, mse, rmse, r2)
                except Exception as e:
                    logging.error("Error calculating metrics: %s", e)
                    return

                plots_dir = os.path.join(epoch_dir, 'plots')
                os.makedirs(plots_dir, exist_ok=True)

                try:
                    logging.debug("Saving loss plot...")
                    plt.figure(figsize=(10, 6))
                    plt.plot(self.train_losses, label='Train Loss', color='blue')
                    plt.plot(self.val_losses, label='Validation Loss', color='orange')
                    plt.title(f'Model Loss up to Epoch {epoch+1}')
                    plt.ylabel('Loss')
                    plt.xlabel('Epoch')
                    plt.legend(loc='upper right')
                    plt.grid(True)
                    loss_plot_path = os.path.join(plots_dir, 'loss_plot.png')
                    plt.savefig(loss_plot_path)
                    plt.close()
                    logging.info("Loss plot saved at %s", loss_plot_path)
                except Exception as e:
                    logging.error("An error occurred while saving loss plot: %s", e)

                try:
                    logging.debug("Saving prediction plot...")
                    evaluator = Evaluator(
                        model=self.model,
                        X_test=self.X_val,
                        y_test=self.y_val,
                        scaler_y=self.main_model_instance.data_loader.scaler_y,
                        run_dir=plots_dir
                    )
                    evaluator.predict()
                    fig_pred = evaluator.plot_predictions()
                    if fig_pred is None:
                        logging.error("Prediction figure is None.")
                        return
                    prediction_plot_path = os.path.join(plots_dir, 'prediction_plot.png')
                    fig_pred.savefig(prediction_plot_path)
                    plt.close(fig_pred)
                    logging.info("Prediction plot saved at %s", prediction_plot_path)
                except Exception as e:
                    logging.error("An error occurred while saving prediction plot: %s", e)

                try:
                    logging.debug("Saving error distribution plot...")
                    fig_error = evaluator.plot_error_distribution()
                    if fig_error is None:
                        logging.error("Error distribution figure is None.")
                        return
                    error_distribution_plot_path = os.path.join(plots_dir, 'error_distribution_plot.png')
                    fig_error.savefig(error_distribution_plot_path)
                    plt.close(fig_error)
                    logging.info("Error distribution plot saved at %s", error_distribution_plot_path)
                except Exception as e:
                    logging.error("An error occurred while saving error distribution plot: %s", e)

                try:
                    if np.isnan(r2):
                        logging.error("R² score is NaN.")
                        return
                    logging.debug("Saving R² plot...")
                    evaluator.plot_r2_bar(r2)
                    logging.info("R² plot saved.")
                except Exception as e:
                    logging.error("An error occurred while saving R² plot: %s", e)

                logging.info("All plots for epoch %d saved in %s.", epoch + 1, plots_dir)
            except Exception as e:
                logging.error(f"An error occurred while plotting: {e}")
