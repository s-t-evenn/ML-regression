import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import torch.nn as nn

# Define the model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)

# Load the scaler
scaler_filename = 'scaler.pkl'
scaler = joblib.load(scaler_filename)

# Load the model
model = torch.load("ML_related/result.pth")
model.eval()  # Set the model to evaluation mode

# Predict given data
#types = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'CabinLetter']
given = [3, 2, 4, 1, 1, 16.7, 1, 1]
given = np.array(given)
given = given.reshape(1, -1)
given = scaler.transform(given)
given = torch.from_numpy(given).float()
prediction = model(given)

print(f"Given: {prediction.item():.2f}")
print("likely to survive") if prediction.item() > 0.5 else print("probebly will not survive")