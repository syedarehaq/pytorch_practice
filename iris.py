# %%
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torchmetrics
from sklearn.preprocessing import StandardScaler
# %%
df_iris = pd.read_csv("./iris.csv")
LE = LabelEncoder()
sc = StandardScaler()
df_iris['class'] = LE.fit_transform(df_iris['variety'])
# %%
features = df_iris[df_iris.columns[0:-2]]
num_features  = features.shape[1]
X = np.array(features, dtype=float)
print(X)
sc.fit(X)
X_std = torch.tensor(sc.transform(X))
print(X_std)
target = df_iris[df_iris.columns[-1]]
y = torch.tensor(np.array(target,dtype=float))
print(y)
num_classes = len(LE.classes_)
# %%
# %%
## Dividing into train test validation
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2
# %%
## Defining the dataset and dataloader
train_dataset = TensorDataset(X_train.float(),y_train.float())
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
validation_dataset = TensorDataset(X_val.float(),y_val.float())
validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True)
# %%
F.one_hot(y.to(torch.int64), num_classes)
# %%
model = nn.Sequential(
    nn.Linear(num_features,10),
    nn.Linear(10,10),
    nn.Linear(10,num_classes)
)
# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
# %%
num_epochs = 500
for epoch in range(num_epochs):
    print(f"{epoch=}")
    training_loss = 0.0
    for data in train_dataloader:
        optimizer.zero_grad()
        feature, target = data
        #print(data)
        pred = model(feature)
        #one_hot_target= F.one_hot(target.to(torch.int64),num_classes)
        loss = criterion(pred,target.long())
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss_training = training_loss/len(train_dataloader)
    print(f"{epoch_loss_training=:0.3f}")

    validation_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i,data in enumerate(validation_dataloader,0):
            feature,target = data
            pred = model(feature)
            loss = criterion(pred,target.long())
            validation_loss += loss.item()
    epoch_loss_validation = validation_loss / len(validation_dataloader)
    print(f"{epoch_loss_validation=:0.3f}")

    metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    for i,data in enumerate(train_dataloader,0):
        features,labels = data
        outputs = model(features)
        acc = metric(outputs, labels)
    acc = metric.compute()
    print(f"Accuracy on all data: {acc}")
    metric.reset()
# %%
test_pred = model(X_test.float())
test_pred_class = test_pred.argmax(dim=-1)
# %%
