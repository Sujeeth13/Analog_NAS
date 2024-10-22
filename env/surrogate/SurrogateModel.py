import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error  
from sklearn.model_selection import train_test_split

# Need to convert Conv Block in the way surrogate model can understand
class SurrogateModel:
    def __init__(self, load_path = None):

        # Load the model
        self.model = None
        if load_path is not None:
            self.load_model(load_path)

        # Define the min and max values for each column
        self.min_values = torch.tensor(
            [8, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            dtype=torch.float32,
        )
        self.max_values = torch.tensor(
            [
                128,
                5,
                16,
                16,
                16,
                16,
                16,
                4,
                4,
                12,
                4,
                4,
                12,
                4,
                4,
                12,
                4,
                4,
                12,
                4,
                4,
                12,
            ],
            dtype=torch.float32,
        )
        self.column_names = [
            "out_channel0",
            "M",
            "R1",
            "R2",
            "R3",
            "R4",
            "R5",
            "convblock1",
            "widenfact1",
            "B1",
            "convblock2",
            "widenfact2",
            "B2",
            "convblock3",
            "widenfact3",
            "B3",
            "convblock4",
            "widenfact4",
            "B4",
            "convblock5",
            "widenfact5",
            "B5",
        ]
        self.value_mapping = {1.0: "A", 2.0: "B", 3.0: "C", 4.0: "D"}
        assert len(self.min_values) == 22
        assert len(self.max_values) == 22
    
    def load_model(self, load_path):
        self.model = xgb.Booster()
        self.model.load_model(load_path)
        print("Surrogate model loaded from: ", load_path)

    
    def train_model(self, 
                    data_path = '../../data/dataset_cifar10_v1.csv', 
                    save_path = '../models/surrogate_model2.json', 
                    params = None, 
                    num_round = 150):
        # Load the data
        data = pd.read_csv(data_path)
        print("Data loaded from: ", data_path)

        # Convert the ConvBlock columns to categorical
        data["convblock1"] = data["convblock1"].astype("category")
        data["convblock2"] = data["convblock2"].astype("category")
        data["convblock3"] = data["convblock3"].astype("category")
        data["convblock4"] = data["convblock4"].astype("category")
        data["convblock5"] = data["convblock5"].astype("category")

        # Split the data into features and target
        X = data.iloc[:,:-3]
        y = data['1_day_accuracy']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Convert the data to DMatrix, which is the internal data structure used by XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        # Define the hyperparameters for the model
        if params is None:
            params = {
                'max_depth': 4,
                'eta': 0.1,
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse'
            }

        # Train the model
        self.model = xgb.train(params, dtrain, num_round)

        # Make predictions on the test set
        predictions = self.model.predict(dtest)
        rmse = root_mean_squared_error(y_test, predictions)
        print(f"Root Mean Squared Error: {rmse}")

        # Save the model
        self.model.save_model(save_path)
        print("Surrogate model saved to: ", save_path)

    def evaluate(self, X):

        X = self.clip_values(X)

        # Convert the input to DMatrix, which is the internal data structure used by XGBoost
        dtest = xgb.DMatrix(X, enable_categorical=True)

        # Use the model to make predictions
        predictions = self.model.predict(dtest)
        # print("Predictions: ", predictions)
        return predictions

    def clip_values(self, X):
        # Shape of X = (batch_size, 22)

        # Round all values
        rounded_data = torch.round(X)

        # Now clamp each column individually
        clamped_data = torch.empty_like(rounded_data)
        for i in range(X.shape[1]):
            clamped_data[:, i] = torch.clamp(
                rounded_data[:, i], self.min_values[i], self.max_values[i]
            )

        output = self.correct_data_format(clamped_data)

        return output

    def correct_data_format(self, data):
        data = data.numpy()

        # Create a pandas DataFrame with the numpy array and set the column names
        df = pd.DataFrame(data, columns=self.column_names)

        df["convblock1"] = (
            df["convblock1"].replace(self.value_mapping).astype("category")
        )
        df["convblock2"] = (
            df["convblock2"].replace(self.value_mapping).astype("category")
        )
        df["convblock3"] = (
            df["convblock3"].replace(self.value_mapping).astype("category")
        )
        df["convblock4"] = (
            df["convblock4"].replace(self.value_mapping).astype("category")
        )
        df["convblock5"] = (
            df["convblock5"].replace(self.value_mapping).astype("category")
        )

        return df

if __name__ == '__main__':
    model = SurrogateModel()
    model.train_model()