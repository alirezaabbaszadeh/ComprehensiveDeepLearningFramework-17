import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class Trainer:
    def __init__(self, model, X_train, y_train, X_val, y_val, X_test, y_test, scaler_y, epochs=1, batch_size=16, history_path=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.scaler_y = scaler_y
        self.epochs = epochs
        self.batch_size = batch_size
        self.history_path = history_path
        self.start_epoch = self.load_checkpoint()

    def load_checkpoint(self):
        if self.history_path and os.path.exists(self.history_path):
            epoch_file = os.path.join(self.history_path, 'last_epoch.json')
            model_path = os.path.join(self.history_path, 'last_model.h5')
            if os.path.exists(epoch_file) and os.path.exists(model_path):
                with open(epoch_file, 'r') as f:
                    last_epoch = json.load(f).get('last_epoch', 0)
                self.model.load_weights(model_path)
                print(f"Resuming from epoch {last_epoch + 1}")
                return last_epoch + 1
        return 0

    def save_checkpoint(self, epoch):
        if self.history_path:
            epoch_file = os.path.join(self.history_path, 'last_epoch.json')
            model_path = os.path.join(self.history_path, 'last_model.h5')
            with open(epoch_file, 'w') as f:
                json.dump({'last_epoch': epoch}, f)
            self.model.save(model_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")

    class PlotAndSaveCallback(tf.keras.callbacks.Callback):
        def __init__(self, trainer):
            super(Trainer.PlotAndSaveCallback, self).__init__()
            self.trainer = trainer

        def on_epoch_end(self, epoch, logs=None):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            epoch_dir = os.path.join(self.trainer.history_path, f"epoch_{epoch + 1}_{timestamp}")
            os.makedirs(epoch_dir, exist_ok=True)

            plt.figure(figsize=(12, 6))
            plt.plot(logs.get('loss'), label='Train Loss', color='blue')
            plt.plot(logs.get('val_loss'), label='Validation Loss', color='orange')
            plt.title(f'Model Loss During Training - Epoch {epoch + 1}')
            plt.ylabel('Mean Absolute Error (MAE)')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.grid(True)
            loss_plot_path = os.path.join(epoch_dir, 'loss_plot.png')
            plt.savefig(loss_plot_path)
            plt.close()
            print(f"Loss plot saved for epoch {epoch + 1} to {loss_plot_path}.")

            y_val_pred = self.trainer.model.predict(self.trainer.X_val)
            y_val_pred_rescaled = self.trainer.scaler_y.inverse_transform(y_val_pred)
            y_val_rescaled = self.trainer.scaler_y.inverse_transform(self.trainer.y_val)

            plt.figure(figsize=(12, 6))
            plt.plot(y_val_rescaled, label='Actual', color='green')
            plt.plot(y_val_pred_rescaled, label='Predicted', color='red')
            plt.title(f'Validation Data - Actual vs Predicted - Epoch {epoch + 1}')
            plt.xlabel('Samples')
            plt.ylabel('Price')
            plt.legend(loc='upper right')
            plt.grid(True)
            pred_plot_path = os.path.join(epoch_dir, 'validation_prediction_plot.png')
            plt.savefig(pred_plot_path)
            plt.close()
            print(f"Validation prediction plot saved for epoch {epoch + 1} to {pred_plot_path}.")

            mae = mean_absolute_error(y_val_rescaled, y_val_pred_rescaled)
            mse = mean_squared_error(y_val_rescaled, y_val_pred_rescaled)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_val_rescaled, y_val_pred_rescaled)

            plt.figure(figsize=(10, 6))
            plt.bar(['MAE', 'MSE', 'RMSE', 'R²'], [mae, mse, rmse, r2], color=['blue', 'orange', 'green', 'purple'])
            plt.title(f'Error Metrics - Epoch {epoch + 1}')
            plt.ylabel('Value')
            plt.grid(True, axis='y')
            error_metrics_path = os.path.join(epoch_dir, 'error_metrics_plot.png')
            plt.savefig(error_metrics_path)
            plt.close()
            print(f"Error metrics plot saved for epoch {epoch + 1} to {error_metrics_path}.")

            y_test_pred = self.trainer.model.predict(self.trainer.X_test)
            y_test_pred_rescaled = self.trainer.scaler_y.inverse_transform(y_test_pred)
            y_test_rescaled = self.trainer.scaler_y.inverse_transform(self.trainer.y_test)

            plt.figure(figsize=(12, 6))
            plt.plot(y_test_rescaled, label='Actual', color='green')
            plt.plot(y_test_pred_rescaled, label='Predicted', color='blue')
            plt.title(f'Test Data - Actual vs Predicted')
            plt.xlabel('Samples')
            plt.ylabel('Price')
            plt.legend(loc='upper right')
            plt.grid(True)
            test_plot_path = os.path.join(epoch_dir, 'test_prediction_plot.png')
            plt.savefig(test_plot_path)
            plt.close()
            print(f"Test prediction plot saved to {test_plot_path}.")

            self.trainer.save_checkpoint(epoch)

    def train(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            self.PlotAndSaveCallback(self)
        ]
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.epochs,
            initial_epoch=self.start_epoch,
            batch_size=self.batch_size,
            validation_data=(self.X_val, self.y_val),
            verbose=1,
            callbacks=callbacks
        )
        return self.history
