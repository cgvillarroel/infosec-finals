# import libraries
import pandas as pd
from joblib import load
from random import randrange
from sklearn.preprocessing import MinMaxScaler

# load and prep the data
class Preprocessor:
    def __init__(self, **kwargs):
        data_path = kwargs.get("data_path", "")
        scaler_path = kwargs.get("scaler_path", "")

        if data_path:
            self.load_data(data_path)

        if scaler_path:
            self.load_scaler(scaler_path)

    def load_data(self, data_path: str):
        self.data: pd.DataFrame = pd.read_csv(data_path)
        return self

    def load_scaler(self, scaler_path: str):
        self.scaler = load(scaler_path)
        return self

    def process(self):
        if self.data is None:
            raise TypeError("No data provided (use `Preprocessor.load_data`)")

        if self.scaler is None:
            raise TypeError("No scaler provided (use `Preprocessor.load_scaler`)")

        # prep data
        self.data.columns = self.data.columns.str.strip()
        self.data["Label"] = self.data["Label"].apply(lambda x: 1 if x == "DDoS" else 0)

        # get columns of interest
        self.data = self.data[[
            "Average Packet Size",
            "Avg Bwd Segment Size",
            "Bwd Packet Length Max",
            "Bwd Packet Length Mean",
            "Bwd Packet Length Min",
            "Bwd Packet Length Std",
            "Down/Up Ratio",
            "Max Packet Length",
            "Packet Length Std",
            "Packet Length Variance",
            "Label"
        ]] # type: ignore

        # scale by same amount as training data
        self.scaler: MinMaxScaler = load("./utils/scaler.joblib")
        self.data = pd.DataFrame(
            self.scaler.transform(self.data),
            columns=self.data.columns
        )

        return self.data
