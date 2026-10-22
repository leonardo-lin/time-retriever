# 面向负荷预测的未来感知 RAFT — 开发规格说明

## 0. 文档状态

- **项目类型：** 研究原型 / 学位论文实验
- **主要目标：** 基于 RAFT 范式，开发并评估一套未来感知的检索增强负荷预测框架。
- **核心假设：** 历史样本的检索应当依据其对预测未来的有用程度，而不仅依据已观测历史负荷形态之间的相似度。
- **当前实现目标：** 先做确定性点负荷预测；检索流程验证通过后，再加入概率预测。
- **重要约束：** 真实未来目标 `Y` 在推理时绝不能提供给检索器。`Y` 与候选未来轨迹 `Yi` 只能用于构造训练监督，以及训练/评估预测生成器。

---

# 1. 研究目标

## 1.1 问题定义

给定已观测的历史窗口：

X = [x_{t-L+1}, ..., x_t]

预测未来负荷：

Y = [x_{t+1}, ..., x_{t+H}]

其中：

- `L` = 输入/历史长度
- `H` = 预测时域（forecast horizon）

所提框架从历史数据库中检索历史预测案例：

(X_i, Y_i)

并将其作为预测当前目标的额外证据：

\hat{Y} = G(X, X_i,Y_i_{i=1}^{K})

其中：

- `X` = 当前已观测上下文
- `Xi` = 历史候选上下文
- `Yi` = 紧随 `Xi` 之后的已知未来轨迹
- `K` = 检索案例数量
- `G` = 预测生成器/模型

---



# 2. 研究假设



## H1 — 检索收益

使用检索到的历史案例，应当优于仅接收当前上下文的预测模型。

## H2 — 传统相似度的局限

仅按已观测历史形态的相似度检索案例，并不必然检索到对预测最有用的样本。

## H3 — 未来感知检索

以未来轨迹信息作为监督来训练检索器，可以学会识别对预测更有用的历史案例。

## H4 — 过去不同、未来相似

历史过去形态差异很大、但未来轨迹相似的案例，仍然可以是有用的检索样本。

## H5 — 难负例

历史过去形态高度相似、但未来轨迹差异很大的案例，是特别有用的难负例，因为它们暴露了朴素“按过去相似度检索”的弱点。

---



# 3. 整体系统

完整的所提系统如下：

```text
Historical Dataset
      │
      ├────────────── Historical Cases ──────────────┐
      │                                               │
      │                                        (Xi, Yi)
      │                                               │
      ↓                                               │
Current X                                              │
      │                                               │
      ↓                                               │
Future-Aware Retriever                                 │
      │                                               │
      ↓                                               │
Top-K historical candidates ──────────────────────────┘
      │
      ↓
Retrieved {(Xi, Yi)}
      │
      ↓
Generator / Forecasting Model
      │
      ↓
Predicted Future Load Ŷ
```

训练阶段还会使用已知的当前未来 `Y`：

```text
Current:
X ─────────→ Y

Candidate:
Xi ────────→ Yi

X, Xi ─────────→ Retriever score
Y, Yi ─────────→ Future relevance supervision
```

未来目标 `Y` 绝不是推理时检索器的输入。

---



# 4. 数据表示



## 4.1 预测样本

每个样本必须包含：

```text
sample_id
timestamp
X
Y
optional exogenous features
```

概念上：

```text
X:
[t-L+1, ..., t]

Y:
[t+1, ..., t+H]
```



## 4.2 建议的初始配置

从以下配置开始：

- 历史长度：`168 hours`
- 预测时域：`24 hours`
- 采样频率：`hourly`

对应关系为：

```text
7 days observed
→
24 hours forecast
```

这些取值必须保持可配置。

配置示例：

```yaml
history_length: 168
forecast_horizon: 24
frequency: 1H
```

不要在模型代码中硬编码这些值。

---



# 5. 输入特征



## 5.1 必需的基线特征

至少包括：

```text
load
```



## 5.2 可选外生特征

框架应支持：

```text
temperature
humidity
weather variables
hour of day
day of week
month
holiday indicator
solar generation
economic variables
other available grid variables
```

这些特征应可配置，而不是永久绑定在模型上。

## 5.3 特征分组

保持独立的特征组：

```text
load_features
calendar_features
weather_features
other_exogenous_features
```

这样便于做消融实验。

---



# 6. 数据集构建



## 6.1 滑动窗口

对每个时间戳 `t`：

```text
Past:
[t-L+1, ..., t]

Future:
[t+1, ..., t+H]
```

构造一个预测样本。

示例：

```text
2025-01-08 00:00 → 2025-01-14 23:00
                    ↓
2025-01-15 00:00 → 2025-01-15 23:00
```



## 6.2 历史候选的定义

每个合格的历史时间戳 `ti` 生成：

```text
Xi = data[ti-L+1 : ti]
Yi = data[ti+1 : ti+H]
```

这一对：

```text
(Xi, Yi)
```

就是一个可检索的历史预测案例。

## 6.3 防止时间泄漏

候选案例必须满足：

```text
candidate_future_end < current_sample_start
```

以保证严格的历史检索。

绝不允许：

```text candidate X or Y

```

包含当前目标时段的信息。

若当前目标从 `T` 开始，则候选数据必须在 `T` 之前结束。

---



# 7. 数据集划分

使用按时间顺序的划分。

建议：

```text
Train
───────|──────────────
       |
Validation
       |──────────────
                    |
Test
                    |──────────────
```

不要在训练/验证/测试之间随机打乱时间戳。

示例：

```yaml
train: 70%
validation: 15%
test: 15%
split_method: chronological
```

具体比例可配置。

## 7.1 检索数据库

每次实验都必须单独定义检索数据库。

默认：

```text
Training period → retrieval database for training
Training + validation period → retrieval database for validation/test only if explicitly allowed by the experimental protocol
```

若要最干净地评估泛化能力，默认测试协议应当只从每个测试查询发生之前已经存在的历史数据中检索。

---



# 8. 归一化

负荷数值在不同时段可能差异很大。

实现必须支持：

### 方案 A — 全局归一化

x' = \frac{x-\mu_{train}}{\sigma_{train}}

统计量只根据训练数据计算。

### 方案 B — 窗口归一化

对每个窗口：

x'_t = \frac{x_t-\mu_X}{\sigma_X}

这种做法强调形态相似度。

### 建议实验

比较：

1. 全局归一化
2. 按窗口 z-score
3. 原始负荷 + 尺度信息

不要在未经说明的情况下混用不同归一化策略。

---



# 9. 检索单元

基本检索对象为：

```text
HistoricalCase {
    case_id
    end_timestamp
    X
    Y
    metadata
}
```

示例：

```json
{
  "case_id": "2025-01-14T23:00",
  "X": "168-hour historical window",
  "Y": "24-hour future trajectory",
  "metadata": {
    "weekday": 1,
    "month": 1
  }
}
```

---



# 10. 检索器架构



## 10.1 检索器输入

检索器必须接收：

```text
current X
candidate Xi
```

或其编码表示。

推理时：

```text
Retriever(X)
```

用于检索候选案例。

检索器不得接收当前的 `Y`。

## 10.2 编码器

初始实现：

```text
Time-Series Transformer Encoder
```

架构应可配置。

示例：

```yaml
retriever:
  encoder: transformer
  d_model: 256
  num_layers: 4
  num_heads: 8
  dropout: 0.1
```

这些是起步取值，不是固定的研究结论。

## 10.3 嵌入

编码器将每个时间序列窗口转换为：

z_X = Encoder(X)

以及：

z_i = Encoder(X_i)

相似度：

S_p(X,X_i)=cos(z_X,z_i)

初始实现应使用余弦相似度。

其他相似度函数可以稍后评估。

---



# 11. 候选检索



## 11.1 检索数据库

每个历史候选必须包含：

```text
case_id
embedding
X
Y
timestamp
metadata
```



## 11.2 ANN 索引

为了可扩展性，使用 ANN 向量索引。

可选实现：

```text
FAISS
```

初始检索流程：

```text
Query embedding
    ↓
ANN search
    ↓
Top-M candidates
    ↓
Filtering
    ↓
Top-K final candidates
```

其中：

- `M > K`
- `M` 允许在最终选择之前做时间/重复过滤。

示例：

```yaml
retrieval:
  candidate_pool_size: 100
  top_k: 5
```

---



# 12. 候选过滤

接受候选之前：

1. 候选必须在时间上合法。
2. 候选不得与当前目标重叠。
3. 若已配置，候选必须满足最小时间间隔。
4. 应去除重复或近重复窗口。
5. 必须满足缺失数据约束。

---



# 13. 未来感知监督

对每个当前训练样本：

```text Current:
X → Y

Candidate:
Xi → Yi
```

计算：

### 过去相似度

S_p = sim(X,X_i)

### 未来相似度

S_f = sim(Y,Y_i)

具体相似度函数必须可配置。

初始建议：

```text normalized MSE
+
correlation
```

例如：

D_f(Y,Y_i)=MSE(normalize(Y),normalize(Y_i))

并将距离转换为相关度：

S_f = \exp(-D_f/\tau)

其中 `tau` 是温度参数。

---



# 14. 正例 / 负例分类

使用二维分类。


| 过去相似度 | 未来相似度 | 类别              |
| ----- | ----- | --------------- |
| 高     | 高     | Strong Positive |
| 高     | 低     | Hard Negative   |
| 低     | 高     | Future Positive |
| 低     | 低     | Easy Negative   |


要点：

### Strong Positive（强正例）

历史案例的过去相似，且未来有用。

### Hard Negative（难负例）

从已观测过去看很相似，但未来不相似。

这一点尤其重要。

### Future Positive（未来正例）

历史案例的过去不同，但未来相似。

这类样本不应被自动当作负例。

### Easy Negative（易负例）

历史案例的过去和未来都不同。

---



# 15. 阈值

不要在未经验证的情况下硬编码任意阈值。

初始实现可以使用分位数。

例如：

```text high similarity = top 20%
low similarity = bottom 50%
```

或者：

```yaml
positive_future_quantile: 0.8
negative_future_quantile: 0.2
```

阈值必须只根据训练集确定。

之后再做敏感性分析。

---



# 16. 检索器训练目标

建议分两个阶段。

## 阶段 1 — 简单对比学习基线

使用：

```text
X
positive Xi
negative Xi
```

以及 InfoNCE 风格的损失：

L_{retriever}

-\log
\frac{
\exp(sim(z_X,z_{pos})/\tau)
}{
\exp(sim(z_X,z_{pos})/\tau)
+
\sum_j \exp(sim(z_X,z_{neg,j})/\tau)
}

这提供一条清晰的基线。

## 阶段 2 — 未来感知排序

把未来相似度作为相关度信号。

期望行为：

R(X,X_a)>R(X,X_b)

若：

S_f(Y,Y_a)>S_f(Y,Y_b)

可以使用排序损失：

L_{rank}

\max(0,m-R(X,X^+)+R(X,X^-))

其中：

- `m` = 间隔（margin）
- `X+` = 更有用的历史候选
- `X-` = 较不有用的候选

---



# 17. 建议的检索器损失

可以实现组合目标：

L_R =
\lambda_{con}L_{contrastive}
+
\lambda_{rank}L_{ranking}

起步取值：

```yaml
lambda_contrastive: 1.0
lambda_ranking: 1.0
```

并做消融。

不要假定这个组合损失就是最优的。

---



# 18. 生成器



## 18.1 生成器输入

生成器接收：

```text
Current X
+
Retrieved historical cases
```

初始所提表示：

G(X,X_i,Y_i_{i=1}^{K})
\rightarrow \hat{Y}

因此每个检索样本包含：

```text
Xi
Yi
```



## 18.2 为什么要包含 Xi？

`Xi` 提供历史上下文，说明 `Yi` 是在什么条件下发生的。

但这必须经过实验检验。

消融：

```text
A: Current X only
B: Current X + retrieved Yi
C: Current X + retrieved Xi + Yi
```

---



# 19. 生成器架构

初始模型：

```text
Transformer-based forecasting model
```

可能的结构：

```text
Current X
    ↓
Encoder
    ↓
Current representation

Retrieved Xi,Yi
    ↓
Retrieved-case encoder
    ↓
Retrieved representations

Current representation
        +
Retrieved representations
        ↓
Fusion / Cross-Attention
        ↓
Forecast Decoder
        ↓
Ŷ
```

具体架构应保持模块化。

---



# 20. 生成器损失



### MSE

L_{MSE}

\frac{1}{H}
\sum_{h=1}^{H}(Y_h-\hat{Y}_h)^2

初始训练可以使用 MSE。

评估指标：

```text MAE
RMSE
MAPE
```

如果负荷可能为零或接近零，不要直接针对 MAPE 优化。

---

# 22. 防止数据泄漏

本节为强制要求。

基本上只要以2025冬季到最新的資料（2026資料）即可解決洩漏問題

## 检索器

推理时，检索器只接收预测时刻已经可用的信息。

允许：

```text past load
known calendar features
known weather information
```

不允许：

```text actual future load Y
future load-derived statistics
future target-derived labels
```



## 生成器

生成器可以接收检索到的 `Yi`，因为 `Yi` 属于历史案例。

生成器绝不能接收当前目标 `Y`。

## 训练标签

`Y` 可以用于：

```text calculate future similarity
construct positive/negative labels
calculate forecasting loss
```

这是允许的，因为它属于训练监督。

---



# 23. 重要的泄漏边界情况

若候选的未来 `Yi` 与当前样本的目标时段重叠，则拒绝该候选。

对当前目标：

```text T+1 ... T+H

```

候选必须满足：

```text candidate_future_end < T+1

```

这一点必须在代码中显式检查。

---



# 24. 基线实验

至少实现：

## 基线 0 — 无检索

```text
X → Forecasting Model → Y
```



## 基线 1 — 随机检索

```text
X
+
random historical Xi,Yi
→ Generator
```

用于检验检索本身是否有用。

## 基线 2 — 按过去相似度检索

```text
X
↓
similarity(X,Xi)
↓
Top-K
↓
Generator
```

这是传统 RAFT 风格的基线。

## 基线 3 — Oracle 未来检索

仅用于训练/评估诊断：

```text
Y
↓
similarity(Y,Yi)
↓
Top-K
```

不得将其表述为可部署的推理方法。

其目的是估计未来感知检索质量的上界。

## 所提方法 — 学习得到的未来感知检索器

```text
X
↓
learned Retriever
↓
Top-K useful historical cases
↓
Generator
```

训练使用未来感知监督。

---



# 25. 建议的消融实验



## 检索信息

```text A: X only
 B: X + calendar
 C: X + weather
 D: X + calendar + weather
```



## 检索内容

```text A: Yi only
B: Xi + Yi
```



## 检索策略

```text A: Random
B: Past similarity
C: Learned retriever
D: Future-aware learned retriever
```



## 检索案例数量

```text K = 1
K = 3
K = 5
K = 10
K = 20
```



## 难负例策略

```text A: Random negatives
B: ANN hard negatives
C: Past-similar / future-dissimilar negatives
D: Mixed negatives
```

---



# 26. 主实验表

最终主结果表应类似：


| 方法                   | 检索方式            | 未来感知训练 | MAE | RMSE | MAPE |
| -------------------- | --------------- | ------ | --- | ---- | ---- |
| Forecasting Baseline | None            | No     |     |      |      |
| Random Retrieval     | Random          | No     |     |      |      |
| RAFT-style           | Past similarity | No     |     |      |      |
| Learned Retriever    | Learned         | No     |     |      |      |
| Proposed             | Future-aware    | Yes    |     |      |      |


---



# 27. 检索评估

不要只评估最终预测指标。

也要评估检索本身。

## Recall@K

若已定义真实相关的历史案例：

Recall@K =
\frac{ relevant\ cases\ retrieved\ in\ top-K}
{ relevant\ cases}

## 检索案例的未来相似度

报告：

```text average Future Similarity @ K

```

比较：

```text random
past similarity
proposed retriever
```



## 难负例检索率

衡量检索器选出下列情况的频率：

```text high Past Similarity
+
low Future Similarity
```

成功的检索器应当减少这类情况。

---



# 28. 按时域的预测评估

不要只报告一个汇总数字。

报告：

```text
1–6 hours
7–12 hours
13–18 hours
19–24 hours
```

或其他合适的时域分组。

原因：

检索对长时域预测的帮助，可能与对短时域预测的帮助不同。

---



# 29. 按工况的预测评估

若数据允许，分别评估：

```text weekday
weekend
holiday
summer
winter
high-load periods
low-load periods
extreme weather
```

这可以揭示检索在异常工况下是否特别有用。

---



# 30. 可视化要求

至少生成：

## 30.1 预测图

```text
Actual Y
Predicted Ŷ
```



## 30.2 检索案例

绘制：

```text
Current X
Retrieved X1, X2, ..., XK
```

以及另外：

```text
Actual Y
Retrieved Y1, Y2, ..., YK
Predicted Ŷ
```



## 30.3 检索二维分析

用以下坐标绘制候选：

```text x-axis = Past Similarity
y-axis = Future Similarity
```

颜色/标签：

```text Strong Positive
Future Positive
Hard Negative
Easy Negative
```

这张图对验证所提研究想法尤其重要。

---



# 31. 数据流水线

建议流水线：

```text
Raw Data
   ↓
Data Cleaning
   ↓
Resampling
   ↓
Missing Value Handling
   ↓
Feature Construction
   ↓
Chronological Split
   ↓
Window Generation
   ↓
Normalization
   ↓
Historical Case Database
   ↓
Retriever Training
   ↓
ANN Index
   ↓
Retrieval
   ↓
Generator Training
   ↓
Evaluation
```

每个阶段都应保存可复现的中间元数据。

---



# 32. 缺失数据

定义明确规则。

例如：

```yaml
max_missing_ratio: 0.05
```

若缺失比例超过阈值：

```text discard window

```

否则：

```text interpolate / impute

```

插补策略必须写进文档。

绝不允许未来信息进入过去窗口的插补。

---



# 33. 可复现性

每次实验都必须保存：

```text random_seed
dataset_version
train/validation/test date ranges
history_length
forecast_horizon
features
normalization method
retriever architecture
generator architecture
learning rate
batch size
epochs
temperature
negative sampling strategy
top-K
ANN configuration
evaluation metrics
```

建议：

```yaml
seed: 42
```

最终实验使用多个随机种子。

---



# 34. 建议的项目结构

```text
project/
│
├── configs/
│   ├── baseline.yaml
│   ├── raft.yaml
│   └── proposed.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── datasets/
│   ├── window_dataset.py
│   ├── retrieval_dataset.py
│   └── historical_case.py
│
├── models/
│   ├── retriever/
│   │   ├── encoder.py
│   │   ├── similarity.py
│   │   └── loss.py
│   │
│   └── generator/
│       ├── transformer.py
│       └── fusion.py
│
├── retrieval/
│   ├── index.py
│   ├── candidate_mining.py
│   ├── hard_negative.py
│   └── filtering.py
│
├── training/
│   ├── train_retriever.py
│   ├── train_generator.py
│   └── train_joint.py
│
├── evaluation/
│   ├── retrieval_metrics.py
│   ├── forecasting_metrics.py
│   └── visualization.py
│
├── experiments/
│   ├── baseline_no_retrieval/
│   ├── random_retrieval/
│   ├── raft/
│   └── proposed/
│
└── README.md
```

---



# 35. 开发阶段



## 阶段 1 — 数据流水线

实现：

- 加载原始负荷数据
- 清洗
- 重采样
- 按时间顺序划分
- 滑动窗口生成
- 归一化
- 历史案例构建

**交付物：** 可复现的 `(X,Y)` 数据集。

---



## 阶段 2 — 预测基线

实现：

```text
X → Transformer → Y
```

无检索。

**交付物：** 基线 MAE/RMSE/MAPE。

---



## 阶段 3 — 传统 RAFT 基线

实现：

```text
X
↓
Past similarity
↓
Top-K historical Xi
↓
retrieve Yi
↓
Generator
```

**交付物：** 与无检索基线的对比。

---



## 阶段 4 — 检索器

实现：

```text
Transformer encoder
+
cosine similarity
+
ANN retrieval
```

**交付物：** 学习式检索系统。

---



## 阶段 5 — 未来感知监督

实现：

```text
future similarity
+
positive/negative construction
+
hard-negative mining
+
contrastive/ranking loss
```

**交付物：** 未来感知检索器。

---



## 阶段 6 — 生成器集成

实现：

```text
Current X
+
Retrieved Xi,Yi
→
Generator
→
Ŷ
```

**交付物：** 完整的所提框架。

---



## 阶段 7 — 消融

运行：

- 无检索
- 随机检索
- RAFT 风格检索
- 学习式检索
- 未来感知检索
- 不同 K
- 不同特征组
- 仅 Yi 与 Xi+Yi
- 不同负例策略

---



## 阶段 8 — 分析

分析：

- 检索质量
- 预测精度
- 难负例
- 未来正例
- 不同负荷工况
- 预测时域
- 失败案例

---



# 36. 需要排查的关键失败案例

系统必须记录以下例子：

### 案例 A

```text
Past very similar
Future very different
```

潜在难负例。

### 案例 B

```text
Past very different
Future very similar
```

潜在未来正例。

### 案例 C

```text
Retriever selects an apparently similar case
but generator becomes worse
```



### 案例 D

```text
Retrieved cases are nearly duplicates
```



### 案例 E

```text
Retrieved case comes from an inappropriate season/regime
```

这些例子应人工检查。

---



# 37. 重要的研究表述

所提模型不应声称：

> "The model retrieves the most similar historical load."
>
> “模型检索到的是最相似的历史负荷。”

更强、也更准确的表述是：

> **The model learns to retrieve historical forecasting cases that are expected to be useful for future load prediction.**
>
> **模型学会检索那些预期对未来负荷预测有用的历史预测案例。**

论文全文都应保持这一区分。

---



# 38. 建议术语

使用：

### Historical Case（历史案例）

(X_i,Y_i)

而不是简单称为 "document"。

### Query / Current Case（查询 / 当前案例）

X

### Past Similarity（过去相似度）

S_p

### Future Similarity（未来相似度）

S_f

### Future-aware Retrieval（未来感知检索）

用未来轨迹信息作为监督来训练检索。

### Hard Negative（难负例）

过去相似度高，但未来相似度低。

### Future Positive（未来正例）

过去相似度低，但未来相似度高。

### Retrieved Future（检索到的未来）

与检索到的历史案例相关联的 `Yi`。

---



# 39. 最终数学定义

当前预测任务：

Y=f(X)

标准预测：

\hat{Y}=G(X)

RAFT 风格预测：

\hat{Y}

G(X,\mathcal{R}(X))

其中：

\mathcal{R}(X)

(X_i,Y_i)_{i=1}^{K}

所提未来感知检索器使用下式训练：

R_\theta(X,X_i)

同时由未来轨迹提供监督：

S_f(Y,Y_i)

期望关系为：

R_\theta(X,X_i)
\approx
f(S_f(Y,Y_i),S_p(X,X_i))

但在推理时：

\boxed{
\mathcal{R}(X)
\text{ uses only information available before the forecast horizon.}
}

框内含义：`\mathcal{R}(X)` 只使用预测时域开始之前已经可用的信息。

---



# 40. 最终研究流水线

```text
                    ┌─────────────────────────┐
                    │    Historical Dataset    │
                    └────────────┬────────────┘
                                 │
                     Build (Xi, Yi) cases
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ↓                                     ↓
       Past Information                        Future Information
              │                                     │
              │                                     │
              ↓                                     ↓
      Retriever Input                        Training Supervision
              │                                     │
              └──────────────────┬──────────────────┘
                                 ↓
                      Future-Aware Retriever
                                 │
                                 ↓
                             Top-K Xi
                                 │
                                 ↓
                       Retrieve corresponding Yi
                                 │
                                 ↓
                       ┌─────────────────┐
                       │    Generator    │
                       │                 │
                       │ Current X       │
                       │       +         │
                       │ Retrieved Xi,Yi │
                       └────────┬────────┘
                                ↓
                         Forecast Ŷ
                                │
                                ↓
                     Compare against Y
                                │
                                ↓
                  MAE / RMSE / MAPE / CRPS
```

---



# 41. 成功标准

若实验能够表明以下几点，则项目成功：

1. 检索相对无检索基线提升了预测效果。
2. 传统的按过去相似度检索会选出难负例。
3. 数据集中存在未来正例，并且它们可以有用。
4. 所提未来感知检索器检索到的案例具有更高的未来相关度。
5. 未来感知检索在预测指标上优于传统 RAFT 风格检索。
6. 在按时间顺序、无泄漏的评估下，该提升仍然成立。
7. 该提升可以通过检索质量分析来解释，而不仅依赖汇总的预测指标。

---



# 42. 首要实现顺序

不要一次实现整个系统。

严格按以下顺序实现：

```text
1. Dataset
   ↓
2. Transformer forecasting baseline
   ↓
3. Historical case database
   ↓
4. Simple past-similarity retrieval
   ↓
5. RAFT-style generator
   ↓
6. Learned retriever
   ↓
7. Future similarity calculation
   ↓
8. Hard-negative / future-positive construction
   ↓
9. Future-aware retriever training
   ↓
10. Full proposed system
   ↓
11. Ablation
   ↓
12. Analysis
```

第一个里程碑应当是：

> **A working RAFT-style baseline on the load dataset before introducing future-aware retriever training.**
>
> **在引入未来感知检索器训练之前，先在负荷数据集上做出可运行的 RAFT 风格基线。**

这样可以保证：所提方法的任何提升，都能归因于检索学习部分，而不是一条尚未验证的端到端流水线。