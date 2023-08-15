import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import joblib

randomnumber = 5201314

# Data preprocessing (turning data into numbers)
dataset = pd.read_csv("ML_related/largerTitanicData.csv")
mean_age = dataset["Age"].mean()

dataset["Age"].fillna(mean_age, inplace=True)
onehotencoder = OneHotEncoder()
onehotencoder.fit(dataset["Embarked"].to_numpy().reshape(-1, 1))
embarkencoded = onehotencoder.transform(dataset["Embarked"].to_numpy().reshape(-1, 1)).toarray()
dataset[['Unknown', 'C', "Q","S"]] = pd.DataFrame(embarkencoded, index=dataset.index)
dataset["Sex"] = dataset["Sex"].map({"male": 1, "female": 2})
dataset["CabinLetter"] = dataset["Cabin"].str[0]
dataset["CabinLetter"] = dataset["CabinLetter"].map({ np.nan: 0,"A" :1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"T":8})
dataset["Fare"] = dataset["Fare"].replace("?", np.nan)
y = dataset["Survived"]
x = dataset.drop(
    ["Survived", "Name", "Ticket", "Cabin", "id", "'boat'", "'body'", "home.dest", "Unnamed: 15", "Unnamed: 16",
     "Cabins", "Aged","Embarked"], axis=1)
for i in x.columns:
    x[i] = x[i].replace(np.nan, 0)
y = y.to_numpy()
x = x.to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.006, random_state=randomnumber)

# Scale the data
scaler = StandardScaler()
scaler.fit(x_train)
scaler_filename = 'scaler.pkl'
joblib.dump(scaler, scaler_filename)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float().view(-1, 1)

# Define the model
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        logits = self.linear(x)
        return torch.sigmoid(logits)

# Initialize the model
input_dim = x_train.shape[1]
model = LogisticRegression(input_dim)
# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.05)

cost_hist = []

# Training loop
for i in range(5000):
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()
    cost_hist.append(loss.item())
    if (i + 1) % 1000 == 0:
        print(f'iteration: {i + 1:4}, loss: {loss.item():.3e}')

# Print the learned weights and biases
weights = model.linear.weight.detach().numpy().flatten()
bias = model.linear.bias.detach().numpy()
print(f'W: {weights}, b: {bias}')

x_test = torch.from_numpy(x_test).float()
y_test_pred = model(x_test)
y_test_pred_class = torch.round(y_test_pred).detach().numpy().flatten()

# A list that shows the id, predicted value, and true value
_, ids = train_test_split(dataset["id"], test_size=0.006, random_state=randomnumber)
print(f'y_test_pred: {y_test_pred_class}')
print(f'y_test: {y_test}')
print("id\ty_test_pred\t y_test")
for id, pred, true in zip(ids, y_test_pred_class, y_test):
    print(f'{id}\t\t{pred}\t\t {true}')
print(f"{criterion(y_test_pred, torch.from_numpy(y_test).float().view(-1, 1)):.3e}")

# Save the model
torch.save(model, "ML_related/result.pth")

#types = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'CabinLetter','Embark(0:Unknown,1:C,2:Q,3:S)']
# Predict given data
given = [3, 2, 4, 1, 1, 16.7, 1,0,1,0,0]
given = np.array(given)
given = given.reshape(1, -1)
given = scaler.transform(given)
given = torch.from_numpy(given).float()
given = model(given)
print(f"Given: {given.item():.2f}")

# Make a graph of cost
import matplotlib.pyplot as plt

# Create a figure object
fig = plt.figure()
fig.patch.set_facecolor('lightgray')
plt.plot(cost_hist)
plt.show()
