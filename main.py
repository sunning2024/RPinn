import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time

# 超参数配置（更新）
config = {
    "input_size": 15,
    "output_size": 1,
    "hidden_size": 128,  # 增大隐藏层大小
    "num_layers": 2,
    "physics_weight": 0.3,  # 调整物理损失权重
    "batch_size": 32,
    "lr": 3e-4,          # 增大学习率
    "epochs": 2000,       # 增加训练轮数
    "weight_decay": 1e-4 # 权重衰减
}

# 数据预处理类（修正标准化）
class MaglevDataset(Dataset):
    def __init__(self, gap_path, current_path, delay_steps=15, 
                 scaler_gap=None, scaler_current=None, smoothing_window=5):
        gap_df = pd.read_csv(gap_path, sep='\s+', header=None, names=['time', 'gap'])
        current_df = pd.read_csv(current_path, sep='\s+', header=None, names=['time', 'current'])
        
        # 数据平滑处理
        self.gap = pd.Series(gap_df['gap']).rolling(
            smoothing_window, min_periods=1, center=True).mean().values.astype(np.float32)
        self.current = pd.Series(current_df['current']).rolling(
            smoothing_window, min_periods=1, center=True).mean().values.astype(np.float32)
        
        # 构建序列数据
        X, y = [], []
        for i in range(len(self.gap)-delay_steps-1):
            X.append(self.gap[i:i+delay_steps])
            y.append(self.current[i+delay_steps])
        
        # 标准化处理
        if scaler_gap is None:
            self.scaler_gap = StandardScaler()
            X = self.scaler_gap.fit_transform(np.array(X))
        else:
            self.scaler_gap = scaler_gap
            X = self.scaler_gap.transform(np.array(X))
            
        if scaler_current is None:
            self.scaler_current = StandardScaler()
            y = self.scaler_current.fit_transform(np.array(y).reshape(-1, 1))
        else:
            self.scaler_current = scaler_current
            y = self.scaler_current.transform(np.array(y).reshape(-1, 1))
        
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# PINN模型（增加网络容量并约束k）
class PhysicsInformedNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            batch_first=True,
            dropout=0.2 if config["num_layers"] > 1 else 0  # 添加dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(config["hidden_size"], 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, config["output_size"])
        )
        self.k_param = nn.Parameter(torch.tensor(0.0))  # 通过exp保证正数
        self.mass = 30000.0

    def forward(self, x, gap_hist):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        pred_current = self.fc(lstm_out[:, -1, :])
        
        if self.training:
            k = torch.exp(self.k_param)  # 确保k为正
            gap_current = gap_hist[:, -1].view(-1, 1)
            F_pred = k * (pred_current**2) / (gap_current**2 + 1e-4)  # 增加分母稳定性
            acc_pred = F_pred / self.mass
            return pred_current, acc_pred
        else:
            return pred_current

# 改进的物理损失函数（鲁棒二阶导数计算）
def physics_loss(acc_pred, gap_hist):
    dt = 0.1
    batch_size, seq_len = gap_hist.shape
    ddg_list = []
    
    # 使用中心差分法计算二阶导数
    for i in range(2, seq_len-2):
        g_prev2 = gap_hist[:, i-2]
        g_prev1 = gap_hist[:, i-1]
        g_next1 = gap_hist[:, i+1]
        g_next2 = gap_hist[:, i+2]
        ddg = (-g_prev2 + 16*g_prev1 - 30*gap_hist[:,i] + 16*g_next1 - g_next2) / (12 * dt**2)
        ddg_list.append(ddg.unsqueeze(1))
    
    if not ddg_list:
        return torch.tensor(0.0, device=acc_pred.device)
    
    ddg = torch.mean(torch.cat(ddg_list, dim=1), dim=1)
    return torch.mean((acc_pred.squeeze() - ddg)**2)

# 训练函数（添加梯度裁剪）
def train(model, train_loader, optimizer, config):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        pred_current, acc_pred = model(x, x)
        
        data_loss = nn.MSELoss()(pred_current, y)
        phys_loss = physics_loss(acc_pred, x)
        loss = data_loss + config["physics_weight"] * phys_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 测试函数
def test(model, test_loader, scaler_current, smoothing_factor=0.05, scale_factor=0.05):
    model.eval()
    preds, truths = [], []
    start_time = time.time()  # 开始计时
    
    with torch.no_grad():
        for x, y in test_loader:
            pred_current = model(x,x)
            preds.append(scaler_current.inverse_transform(pred_current.numpy()))
            truths.append(scaler_current.inverse_transform(y.numpy()))
    
    end_time = time.time()  # 结束计时
    test_time = end_time - start_time
    print(f"Test time: {test_time:.2f} seconds")
    
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    
    # 评估指标
    mae = np.mean(np.abs(preds - truths))
    rmse = np.sqrt(np.mean((preds - truths)**2))
    print(f"Test MAE: {mae:.4f} A, RMSE: {rmse:.4f} A")
    
    # 计算误差
    errors = np.abs(preds - truths)
    
    # 平滑误差曲线 (使用指数移动平均)
    smoothed_errors = np.zeros_like(errors)
    smoothed_errors[0] = errors[0]
    
    for i in range(1, len(errors)):
        smoothed_errors[i] = (1 - smoothing_factor) * smoothed_errors[i - 1] + smoothing_factor * errors[i]
    
    # 对误差进行缩放处理
    scaled_errors = smoothed_errors * scale_factor

    # 绘制图像
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制预测值与真实值
    ax1.plot(truths, color='r', label='True Current')
    ax1.plot(preds, color='b', label='Predicted Current')
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Current (A)")
    ax1.legend(loc='upper left')

    # 创建第二个 y 轴并绘制平滑缩放后的误差曲线
    ax2 = ax1.twinx()
    ax2.plot(scaled_errors, color='g', label='Smoothed Error (Scaled)', linestyle='dotted')
    ax2.set_ylabel("Scaled Error")
    ax2.set_ylim(0, 1)  # 设置误差曲线的 y 轴范围为 0 到 10
    ax2.legend(loc='upper right')

    plt.title('Prediction vs True Values with Smoothed & Scaled Error Curve')
    plt.savefig('res.png')  # 保存图像
    plt.close()  # 关闭图像，防止显示

if __name__ == "__main__":
    # 训练数据（应用平滑）
    train_dataset = MaglevDataset("gap.txt", "I1.txt", smoothing_window=5)
    scaler_gap = train_dataset.scaler_gap
    scaler_current = train_dataset.scaler_current
    
    # 测试数据（使用训练集的scaler）
    test_dataset = MaglevDataset("gap1.txt", "I2.txt", 
                               scaler_gap=scaler_gap,
                               scaler_current=scaler_current)
    
    # 初始化DataLoader
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # 初始化模型和优化器
    model = PhysicsInformedNN(config)
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=config["lr"], 
                                weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # 训练循环
    for epoch in range(config["epochs"]):
        loss = train(model, train_loader, optimizer, config)
        scheduler.step(loss)  # 动态调整学习率
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    # 测试和保存
    torch.save(model.state_dict(), "model_final.pth")
    test(model, test_loader, test_dataset.scaler_current)