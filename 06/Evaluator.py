import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from typing import Optional, Tuple
import os
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set the desired logging level

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - Evaluator - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Evaluator:
    def __init__(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler_y,
        run_dir: str,
        history_manager: Optional[object] = None
    ):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.scaler_y = scaler_y
        self.run_dir = run_dir
        self.y_pred = None
        self.y_pred_rescaled = None
        self.y_test_rescaled = None
        self.history_manager = history_manager
        logger.debug("Evaluator initialized.")

    def predict(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        logger.debug("Generating predictions.")

        if self.X_test is None or self.y_test is None:
            logger.error("X_test or y_test is None.")
            return None, None

        try:
            self.y_pred = self.model.predict(self.X_test, verbose=0)
            logger.debug("Predictions generated.")
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return None, None

        if self.y_pred is None:
            logger.error("Predictions are None after model prediction.")
            return None, None

        try:
            self.y_pred_rescaled = self.scaler_y.inverse_transform(self.y_pred)
            self.y_test_rescaled = self.scaler_y.inverse_transform(self.y_test)
            logger.debug("Predictions and true labels rescaled.")
        except Exception as e:
            logger.error(f"Error during inverse scaling: {e}")
            return None, None

        return self.y_pred_rescaled, self.y_test_rescaled

    def calculate_metrics(self) -> Optional[Tuple[float, float, float, float]]:
        logger.debug("Calculating evaluation metrics.")

        if self.y_test_rescaled is None or self.y_pred_rescaled is None:
            logger.error("Rescaled predictions or true labels are None.")
            return None

        try:
            mae = mean_absolute_error(self.y_test_rescaled, self.y_pred_rescaled)
            mse = mean_squared_error(self.y_test_rescaled, self.y_pred_rescaled)
            rmse = np.sqrt(mse)
            r2 = r2_score(self.y_test_rescaled, self.y_pred_rescaled)
            logger.debug(f"Metrics calculated: MAE={mae}, MSE={mse}, RMSE={rmse}, R2={r2}.")
            return mae, mse, rmse, r2
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return None

    def plot_loss(self, history: Optional[object] = None) -> Optional[plt.Figure]:
        logger.debug("Plotting loss over epochs.")

        if history is None or not hasattr(history, 'history'):
            logger.error("History object is None or does not have 'history' attribute.")
            return None

        if 'loss' not in history.history or 'val_loss' not in history.history:
            logger.error("Loss keys not found in history.")
            return None

        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Train Loss', color='blue')
        plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
        plt.title('Model Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
        logger.debug("Loss plot created.")
        return plt.gcf()

    def plot_mae(self, history, save_path: str):
        logger.debug("Plotting MAE over epochs.")

        if history is None or not hasattr(history, 'history'):
            logger.error("History object is None or does not have 'history' attribute.")
            return

        if 'mae' not in history.history or 'val_mae' not in history.history:
            logger.error("MAE keys not found in history.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(history.history['mae'], label='Train MAE', color='blue')
        plt.plot(history.history['val_mae'], label='Validation MAE', color='orange')
        plt.title('Mean Absolute Error (MAE) During Training')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"MAE plot saved to {save_path}.")

    def plot_mse(self, history, save_path: str):
        logger.debug("Plotting MSE over epochs.")

        if history is None or not hasattr(history, 'history'):
            logger.error("History object is None or does not have 'history' attribute.")
            return

        if 'mse' not in history.history or 'val_mse' not in history.history:
            logger.error("MSE keys not found in history.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(history.history['mse'], label='Train MSE', color='blue')
        plt.plot(history.history['val_mse'], label='Validation MSE', color='orange')
        plt.title('Mean Squared Error (MSE) During Training')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"MSE plot saved to {save_path}.")

    def plot_rmse(self, history, save_path: str):
        logger.debug("Plotting RMSE over epochs.")

        if history is None or not hasattr(history, 'history'):
            logger.error("History object is None or does not have 'history' attribute.")
            return

        if 'mse' not in history.history or 'val_mse' not in history.history:
            logger.error("MSE keys not found in history for RMSE calculation.")
            return

        plt.figure(figsize=(12, 6))
        plt.plot(np.sqrt(history.history['mse']), label='Train RMSE', color='blue')
        plt.plot(np.sqrt(history.history['val_mse']), label='Validation RMSE', color='orange')
        plt.title('Root Mean Squared Error (RMSE) During Training')
        plt.ylabel('RMSE')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        logger.debug(f"RMSE plot saved to {save_path}.")

    def plot_predictions(self) -> Optional[plt.Figure]:
        logger.debug("Plotting actual vs predicted close prices.")

        if self.y_test_rescaled is None or self.y_pred_rescaled is None:
            logger.error("Rescaled predictions or true labels are None.")
            return None

        if len(self.y_test_rescaled) == 0 or len(self.y_pred_rescaled) == 0:
            logger.error("Rescaled predictions or true labels are empty.")
            return None

        plt.figure(figsize=(14, 7))
        plt.plot(self.y_test_rescaled, label='Actual Close Price', color='blue')
        plt.plot(self.y_pred_rescaled, label='Predicted Close Price', color='red', alpha=0.7)
        plt.title('Actual vs Predicted Close Prices')
        plt.xlabel('Sample')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        logger.debug("Predictions plot created.")
        return plt.gcf()

    def plot_error_distribution(self) -> Optional[plt.Figure]:
        logger.debug("Plotting distribution of prediction errors.")

        if self.y_test_rescaled is None or self.y_pred_rescaled is None:
            logger.error("Rescaled predictions or true labels are None.")
            return None

        errors = self.y_test_rescaled - self.y_pred_rescaled

        if np.isnan(errors).any():
            logger.error("Errors contain NaN values.")
            return None

        plt.figure(figsize=(10, 5))
        plt.hist(errors, bins=50, color='purple', edgecolor='black')
        plt.title('Distribution of Prediction Errors')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True)
        logger.debug("Error distribution plot created.")
        return plt.gcf()

    def print_metrics(self, mae: float, mse: float, rmse: float, r2: float) -> None:
        logger.info(f"Mean Absolute Error (MAE): {mae}")
        logger.info(f"Mean Squared Error (MSE): {mse}")
        logger.info(f"Root Mean Squared Error (RMSE): {rmse}")
        logger.info(f"R² Score: {r2}")

    def plot_r2_bar(self, r2_value):
        logger.debug("Plotting R² score as bar chart.")

        if r2_value is None or np.isnan(r2_value):
            logger.error("R² value is None or NaN.")
            return

        plt.figure(figsize=(8, 6))
        plt.bar(['R² Score'], [r2_value], color='#4682B4', edgecolor='black', width=0.6)
        plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, label='Baseline')
        plt.axhline(y=1.0, color='green', linestyle='--', linewidth=0.8, label='Perfect Score')
        plt.ylim(0, 1.1)
        plt.title('R² Score of the Model', fontsize=14, fontweight='bold')
        plt.ylabel('R² Value', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='upper left', fontsize=10)
        plt.text(0, r2_value + 0.03, f"{r2_value:.2f}", ha='center', fontsize=12, fontweight='bold', color='black')
        plt.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
        save_path = os.path.join(self.run_dir, 'r2_score_plot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.debug(f"R² bar plot saved to {save_path}.")
        logger.info("R² plot displayed and saved successfully.")
