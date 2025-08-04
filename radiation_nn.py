import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


# 读取数据
df = pd.read_csv('/Users/wuduojin/Desktop/grey_rad_data_prep/atmospheric_radiation_dataset.csv')

# 分离输入和输出
X = df[['p', 'T','density']].values
Y = df[['up_flux', 'down_flux']].values

# 输入归一化
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# 输出归一化
scaler_Y = MinMaxScaler()
Y_scaled = scaler_Y.fit_transform(Y)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)

# 创建 DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor), batch_size=32, shuffle=False)

class RadiationCNN(nn.Module):
    def __init__(self, input_dim):
        super(RadiationCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(64, 2)  # 每层预测两个通量: 向上和向下

    def forward(self, x):
        # 输入维度: (batch, features, layers)
        x = x.permute(0, 2, 1)  # 将输入转置为 (batch, channels, length)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.permute(0, 2, 1)  # 转换回 (batch, layers, features)
        x = self.fc(x)  # 每层输出两个通量
        return x

# 初始化模型
model = RadiationCNN(input_dim=8)  # 8 个输入特征
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    
    train_loss /= len(train_loader.dataset)

    # 验证
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            test_loss += loss.item() * X_batch.size(0)
    test_loss /= len(test_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

model.eval()
with torch.no_grad():
    Y_pred_scaled = model(X_test_tensor).numpy()

# 反归一化输出
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 2)).reshape(-1, Y_test.shape[1], 2)
Y_true = scaler_Y.inverse_transform(Y_test.reshape(-1, 2)).reshape(-1, Y_test.shape[1], 2)

# 可视化结果
sample_idx = 0  # 可视化第一个样本
plt.plot(Y_true[sample_idx, :, 0], label="True Upward Flux")
plt.plot(Y_pred[sample_idx, :, 0], label="Pred Upward Flux")
plt.plot(Y_true[sample_idx, :, 1], label="True Downward Flux")
plt.plot(Y_pred[sample_idx, :, 1], label="Pred Downward Flux")
plt.xlabel("Layer Index")
plt.ylabel("Flux")
plt.legend()
plt.show()

