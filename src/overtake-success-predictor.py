import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os


class OvertakeSuccessPredictor:
    def __init__(self, data_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, data_file)

        self.data = pd.read_csv(file_path)
        self.model = GradientBoostingClassifier(random_state=42)

    def preprocess_data(self):
        X = self.data[['SpeedDifference', 'TrackPosition', 'TyreDifference', 'DRSActive']]
        y = self.data['OvertakeSuccess']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate_model(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        report = classification_report(y_test, predictions)
        return accuracy, cm, report

if __name__ == "__main__":
    data_file = "example_overtake_data.csv"
    print(f"Expected file path: {os.path.abspath(data_file)}")

    predictor = OvertakeSuccessPredictor(data_file)
    X_train, X_test, y_train, y_test = predictor.preprocess_data()

    predictor.train_model(X_train, y_train)

    accuracy, cm, report = predictor.evaluate_model(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}\n")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    example_data = pd.DataFrame({
        'SpeedDifference': [5.2, -3.1],
        'TrackPosition': [8, 15],
        'TyreDifference': [0.5, -0.3],
        'DRSActive': [1, 0]
    })
    predictions = predictor.predict(example_data)
    for i, pred in enumerate(predictions):
        print(f"Example {i + 1} predicted overtake success: {'Yes' if pred == 1 else 'No'}")