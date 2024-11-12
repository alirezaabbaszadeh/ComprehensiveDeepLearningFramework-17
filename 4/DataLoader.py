import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, file_path, time_steps=48, split_ratio=0.8):
        self.file_path = file_path
        self.time_steps = time_steps
        self.split_ratio = split_ratio
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        logger.debug("DataLoader initialized with file_path=%s, time_steps=%d, split_ratio=%.2f",
                     self.file_path, self.time_steps, self.split_ratio)
    
    def load_data(self):
        if not os.path.exists(self.file_path):
            logger.error("File not found at %s.", self.file_path)
            raise FileNotFoundError(f"The file at {self.file_path} was not found.")
        logger.info("Loading data from %s", self.file_path)
        data = pd.read_csv(self.file_path)
        
        # Check for NaN or None values
        if data.isnull().values.any():
            logger.warning("Data contains NaN values. Dropping NaN values.")
            data = data.dropna()
        
        logger.info("Data loaded successfully. Data shape: %s", data.shape)
        return data
    
    def preprocess_data(self, data):
        logger.info("Starting data preprocessing")
        X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = data['Close'].values
        
        # Log shapes of initial data
        logger.debug("Initial shape of X: %s, y: %s", X.shape, y.shape)
        
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1))
        logger.info("Data preprocessing completed")
        logger.debug("Shape of X_scaled: %s, y_scaled: %s", X_scaled.shape, y_scaled.shape)
        return X_scaled, y_scaled
    
    def create_sequences(self, X, y):
        logger.info("Creating sequences with time_steps=%d", self.time_steps)
        Xs, ys = [], []
        
        for i in range(len(X) - self.time_steps):
            if y[i + self.time_steps] is not None:
                Xs.append(X[i:i + self.time_steps])
                ys.append(y[i + self.time_steps])
            else:
                logger.warning("None value found at index %d in y.", i + self.time_steps)
        
        Xs = np.array(Xs)
        Xs = np.expand_dims(Xs, axis=-1)
        logger.info("Sequences created")
        logger.debug("Shape of Xs: %s, ys: %s", Xs.shape, np.array(ys).shape)
        return Xs, np.array(ys)
    
    def split_data(self, X, y):
        logger.info("Splitting data with split_ratio=%.2f", self.split_ratio)
        split = int(self.split_ratio * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Log shapes of the split data
        logger.info("Data split completed")
        logger.debug("Shape of X_train: %s, X_test: %s", X_train.shape, X_test.shape)
        logger.debug("Shape of y_train: %s, y_test: %s", y_train.shape, y_test.shape)
        
        if len(X_test) == 0 or len(y_test) == 0:
            logger.warning("Test set is empty; more data is needed.")
        
        return X_train, X_test, y_train, y_test
    
    def get_data(self):
        logger.info("Starting data loading and preprocessing")
        data = self.load_data()
        X_scaled, y_scaled = self.preprocess_data(data)
        X_seq, y_seq = self.create_sequences(X_scaled, y_scaled)
        X_train, X_test, y_train, y_test = self.split_data(X_seq, y_seq)
        
        # Check final shapes of the training and test data
        logger.debug("Final shapes - X_train: %s, X_test: %s, y_train: %s, y_test: %s",
                     X_train.shape, X_test.shape, y_train.shape, y_test.shape)
        
        if X_train.size == 0 or y_train.size == 0:
            logger.error("Training set is empty; please check the data.")
        
        logger.info("Data loading and preprocessing completed")
        return X_train, X_test, y_train, y_test, self.scaler_y
