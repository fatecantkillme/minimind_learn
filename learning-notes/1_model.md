# 模型代码解析

## 参数解析

```python
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        # 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率

```

### 基础参数

`dropout`是训练过程中对神经网络的连接进行随机丢弃，防止过拟合。

[4.6. 暂退法（Dropout） — 动手学深度学习 2.0.0 documentation](https://zh.d2l.ai/chapter_multilayer-perceptrons/dropout.html)

`bos_token_id`: int = 1 Beginning-of-Sequence（序列开始）token 的 ID。

`eos_token_id`: int = 2 End-of-Sequence（序列结束）token 的 ID。

`hidden_act: str` = 'silu' 隐藏层（如 FFN 中间层）使用的激活函数。 $silu:f(x) = x \cdot \frac{1}{1 + e^{-\beta x}}$

[SiLU — PyTorch 2.9 文档 - PyTorch 文档](https://docs.pytorch.ac.cn/docs/stable/generated/torch.nn.SiLU.html)

[arxiv.org/pdf/1710.05941v1](https://arxiv.org/pdf/1710.05941v1)

`hidden_size`: int = 512 模型的隐藏层维度（即词向量、注意力输出、FFN 输入/输出的维度）

`intermediate_size`前馈神经网络（FFN）中间层的维度。 hidden_size → intermediate_size → hidden_size

`max_position_embeddings`: int = 32768

`num_attention_heads`: int = 8 多头注意力机制中的“头数”。

`num_hidden_layers`: int = 8 Transformer 的层数（即堆叠多少个 encoder block）。

` num_key_value_heads:` int = 2 用于 **分组查询注意力（Grouped-Query Attention, GQA）**。

- 标准多头注意力：每个头都有独立的 Q、K、V 投影 → `num_heads` 个 K/V。
- **MQA（Multi-Query Attention）**：所有头共享一组 K/V → `num_key_value_heads = 1`。
- **GQA**：介于两者之间，例如 8 个 Q 头，但只用 2 组 K/V（每组被 4 个 Q 头共享）。

注意不同于原始多头注意力机制,**分组查询注意力机制**的**投影矩阵** $W_k$ $W_v$的维度是$d_{model}\times d_{head}$

[Transformer注意力机制：MHA、MQA与GQA的对比 | Yue Shui 博客](https://syhya.github.io/zh/posts/2025-01-16-group-query-attention/)

`vocab_size:` int = 6400 词表大小（即模型能识别的不同 token 数量）

`rms_norm_eps`: float = 1e-05 RMSNorm（Root Mean Square Layer Normalization）中的稳定项 ε。

[arxiv.org/pdf/1910.07467](https://arxiv.org/pdf/1910.07467)

$\bar{\mathbf{x}} = \frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}}$

`rope_theta`: float = 1000000.0  RoPE（Rotary Position Embedding）的基频参数 θ。

[让研究人员绞尽脑汁的Transformer位置编码 - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/8130)