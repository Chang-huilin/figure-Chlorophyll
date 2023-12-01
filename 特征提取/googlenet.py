import torch
import torch.nn as nn
import torchvision.models as models

# 加载GoogLeNet模型
googlenet = models.googlenet(pretrained=True)

# 修改输出层
# 原始的GoogLeNet的全连接层输出1000个类别，这里修改为一个回归节点
num_features = googlenet.fc.in_features
googlenet.fc = nn.Linear(num_features, 1)  # 修改为一个回归节点

# 定义损失函数，这里使用均方误差
criterion = nn.MSELoss()

# 定义优化器
optimizer = torch.optim.Adam(googlenet.parameters(), lr=0.001)

# 准备回归任务的训练数据（假设你有自己的数据加载逻辑）
# ...

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in dataloader:  # 假设你有一个数据加载器
        optimizer.zero_grad()
        outputs = googlenet(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 在测试数据上评估模型（略）
