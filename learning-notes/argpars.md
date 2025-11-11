# argparse 教程

[argparse](https://docs.python.org/zh-cn/3/howto/argparse.html)是Python标准库中的命令行解析模块

## 基础

```python
import argparse
parser = argparse.ArgumentParser()//实例化
parser.add_argument(````````)//添加参数
args=parser.parse_args()//解析参数
print(args.```)//使用参数
```

## 添加参数

### 位置参数

所谓位置参数就是一个唯一的、必须输入的参数

```python
parser.add_argument("file",help="file locate",type=str)
```

argparse会默认把传递的参数作为字符串，可以使用type指定变量类型

### 可选参数

```python
parser.add_argument("--verbose", help="increase output verbosity",
                    action="store_true")//--verbose后面不需要输入
parser.add_argument("--verbosity", help="increase output verbosity")//--verbose后面需要输入
```

### 短选项

```python
parser.add_argument("-v", "--verbose", help="increase output verbosity",
                    action="store_true")//输入时可以只输入-v
```

```shell
python prog.py -v == python prog.py --verbose
```

### 限制参数

我们可以限制 `--verbosity` 选项可以接受的值

```python
parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
                    help="increase output verbosity")
```