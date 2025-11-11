# SwanLab教程

[SwanLab](https://docs.swanlab.cn/guide_cloud/general/what-is-swanlab.html)是一款**开源、轻量**的 AI 模型训练跟踪与可视化工具，提供了一个**跟踪、记录、比较、和协作实验**的平台。

创建一个SwanLab实验分为3步：

1. 初始化SwanLab
2. 传递一个超参数字典
3. 在你的训练循环中记录指标

## 1. 初始化SwanLab

`swanlab.init()`的作用是初始化一个SwanLab实验，它将启动后台进程以同步和记录数据。
下面的代码片段展示了如何创建一个名为 **cat-dog-classification** 的新SwanLab项目。并为其添加了：

1. **project**：项目名。
2. **experiment_name**：实验名。实验名为当前实验的标识，以帮助您识别此实验。
3. **description**：描述。描述是对实验的详细介绍。

```python
# 导入SwanLab Python库
import swanlab

# 1. 开启一个SwanLab实验
run = swanlab.init(
    project="cat-dog-classification",
    experiment_name="Resnet50",
    description="我的第一个人工智能实验",
)
```

当你初始化SwanLab时，`swanlab.init()`将返回一个对象。
此外，SwanLab会创建一个本地目录（默认名称为“swanlog”），所有日志和文件都保存在其中，并异步传输到 SwanLab 服务器。

## 2. 传递超参数字典

传递超参数字典，例如学习率或模型类型。
你在`config`中传入的字典将被保存并用于后续的实验对比与结果查询。

```python
# 2. 传递一个超参数字典
swanlab.config={"epochs": 20, "learning_rate": 1e-4, "batch_size": 32, "model_type": "CNN"}
```

## 3. 在训练循环中记录指标

在每轮for循环（epoch）中计算准确率与损失值指标，并用`swanlab.log()`将它们记录到SwanLab中。
在默认情况下，当您调用`swanlab.log`时，它会创建一个新的step添加到对应指标的历史数据中，规则是新的step=旧的最大step数+1。
下面的代码示例展示了如何用`swanlab.log()`记录指标：

```python
# 省略了如何设置模型与如何设置数据集的细节

# 设置模型和数据集
model, dataloader = get_model(), get_data()

# 训练循环
for epoch in range(swanlab.config.epochs):
    for batch in dataloader:
        loss, acc = model.train_step()
        # 3. 在你的训练循环中记录指标，用于在仪表盘中进行可视化
        swanlab.log({"acc": acc, "loss": loss})
```

## 4. 用argparse设置

你可以用 `argparse` 对象设置 `config`。`argparse` 是Python标准库（Python >= 3.2）中的一个非常强大的模块，用于从命令行接口（CLI）解析程序参数。这个模块让开发者能够轻松地编写用户友好的命令行界面。

可以直接传递 `argparse` 对象设置 `config`：

```python
import argparse
import swanlab

# 初始化Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=20)
parser.add_argument('--learning-rate', default=0.001)
args = parser.parse_args()

swanlab.init(config=args)
```

等同于 `swanlab.init(config={"epochs": 20, "learning-rate": 0.001})`