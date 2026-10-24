# 生成器程序规格书

**状态：已确认，并已按此实现。**

本文件只规定负荷预测生成器这一段要做什么。检索器怎么训练不在范围内；生成器把已经训好的检索器当冻结查表器用。

---

## 1. 目的

用冻结的未来感知检索器，从历史库取出若干 `(Xi, Yi)`，再根据当前 21 天负荷 `X` 生成接下来 7 天的点预测 `\hat{Y}`。

同一套代码也能关掉检索，只拿 `X` 做预测。这样才看得出检索有没有帮上忙。

训练过程要能在本地网页上实时看到损失和验证误差。

## 2. 不做的事

- 不训练检索器，不联合微调。检索器参数全程冻住。
- 不把当前查询的真实未来 `Y` 送进生成器。
- 不把 Moment 当作生成器骨干。Moment 只在冻结的检索器里，负责把一条窗口压成向量。
- 不做随机检索、过去相似 RAFT、Oracle 预测。那是另几项实验。
- 不做 `Yi-only` 相对 `Xi+Yi` 的消融。这一版固定吃 `Xi` 和 `Yi`。
- 不要天气、日历、概率预测，也不按工作日或季节拆指标。

## 3. 生成器的输入和输出

| 阶段 | 生成器输入 | 生成器输出 | `Y` / `Yi` 的角色 |
| --- | --- | --- | --- |
| 训练（冻结检索） | 当前 `X`，以及预先查出的 `(Xi, Yi)` | `\hat{Y}` | `Y` 只用来算 MSE。`Yi` 是历史库里的已知未来，可以进生成器 |
| 推理 | 同上 | `\hat{Y}` | 当前 `Y` 不出现 |
| 无检索 | 只有当前 `X` | `\hat{Y}` | `Y` 只用来算损失和指标 |

因此：生成器训练和推理都不是「输入 `X` 和当前 `Y`、输出 `\hat{Y}`」。当前 `Y` 只是监督。历史 `Yi` 可以进模型，因为它在预测时刻已经存在于检索库。

检索发生在生成器训练之前，只做一次。训练循环里不再跑检索器。

## 4. 数据

不新写读 CSV、切窗、泄漏检查。直接调用 `retriever.data.load_experiment`，切分规则与检索器规格第 4 节相同。

来源：`data/caiso_demand_forecast/SLD_FCST_ACTUAL.csv`  
只用 `CA ISO-TAC`。时间一律按加州本地时间。每天本地 23:00 取一条。

| 集合 | 规则 |
| --- | --- |
| 检索库 | 案例的 7 天未来必须在 2025-01-01 00:00 之前结束 |
| 训练查询 | 预测起点落在 2025 年，并且后面仍有完整的 7 天 |
| 验证 / 测试 | 预测起点落在 2026 年，随机打乱后按 50% / 50% 切开 |

切分参数从检索器 checkpoint 里的 `config` 还原，不在生成器配置里再抄一份日期。这样生成器看到的检索库和查询，与训这个编码器时用的是同一套。

程序必须检查：检索库里最晚的未来结束时间，早于第一条训练查询的预测起点。`load_experiment` 已经会停；生成器不得绕过这次检查。

一条案例：

```text
预测起点 t = 该案例的“现在”
X = [t-503h, ..., t]          21 天，504 小时
Y = [t+1h, ..., t+168h]       7 天，168 小时
```

验证损失和验证 MAE 仍记录并画在网页上。测试只在训练结束后做预测质量检查，不参与更新参数。模型按验证集 MAE 最低的一轮保存，不用训练 MSE 来选模型。

## 5. 冻结检索与查表

默认 `retrieval_source=frozen`。

1. 必须提供已经训好的 `encoder.pt`。没有就停止，不许训练。
2. 用文件里的 `config` 构造 `RetrieverEncoder`，再 `load_state_dict`。当前默认是本地 `MOMENT-1-large`，主干冻住，T5 的 q/v 带 LoRA。缺权重目录或 `peft` 就停止。
3. 调用现成的 `retriever.evaluate.embed`，分别嵌入查询的 `X` 和检索库的 `Xi`。批大小用检索器配置里的 `eval_batch_size`（默认 8），不要改成大 batch。
4. 点积取 Top-K（默认 5）。得到的是检索库行号，不是负荷。
5. 训练或评估时再按行号查表：`database.x[行号]` 是 `Xi`，`database.y[行号]` 是 `Yi`。

`retrieval_source=none` 时仍读取 `encoder.pt` 里的 `config` 来切数据，但不构造编码器、不算相似度。行号形状是 `(查询数, 0)`，生成器看不到任何 `(Xi, Yi)`。

预计算对训练、验证、测试各做一次，结果存成 `Retrieved`，训练循环只读行号。`none` 时这份对象第二维为 0。

## 6. 模型与损失

生成器是从零训练的拼接 Transformer，不是 Moment。Moment 只会把一条窗口压成向量，融不进 K 条历史案例。

内部用当前 `X` 的均值和标准差做实例标准化，把 `X`、`Xi`、`Yi`、`Y` 都放到当前负荷尺度上。预测在这个标准化空间里做；评估时用同一组统计量反标准化回 MW。

```text
7 个学习出的每日 query token
+ 当前 X 切成 21 个 24 小时 patch
+ 每条检索案例：Xi 的 21 个 patch 取平均，得到 1 个 token
                Yi 切成 7 个 24 小时 patch
→ TransformerEncoder（d_model 256，4 层，8 头）
→ 取出那 7 个每日 token，各自映到 24 小时
→ 168 维标准化 Ŷ
```

`K=0` 时没有检索 token，就是无检索基线。K=5 时序列长度是 `7 + 21 + 5 × 8 = 68`。

解码不用 `21 × 256 → 168` 那种大线性层。查询大约一年 365 条，过大的头容易过拟合。生成器的 256 维与检索器投影的 128 维无关，不要从 checkpoint 抄过来。

训练损失是标准化空间上的 MSE：

```text
L = mean((Y_norm − Ŷ_norm)²)
```

评估用反标准化后的负荷：

```text
MAE、RMSE、MAPE
以及第 1 到第 7 天各自的 MAE
```

全系统负荷以万千瓦计，不会碰到零附近，MAPE 可以报，但不要拿它当训练目标。验证时关闭 dropout，不反传。

## 7. 监控

训练进程自己在 `127.0.0.1:8766` 打开网页，每秒刷新，不依赖外网。端口避开检索器的 8765。

页面必须同时显示：

- 每个 batch 的训练 MSE
- 每一轮结束时的训练 MSE、验证 MSE、验证 MAE
- 检索库、训练、验证、测试各有多少条，以及起止时间
- 这次用的是冻结检索还是无检索，以及 K
- 训练结束后的测试 MAE、RMSE、MAPE，以及 7 天分日 MAE

## 8. 程序分块

程序按这个边界分成这些文件。数据切分和嵌入函数从 `retriever` 进来，不复制。

```text
config      检索器权重路径、retrieval_source、K、模型大小、端口
retrieve    加载冻结编码器，预计算 Top-K 行号
model       ForecastGenerator
train       把上面几块串起来；MSE 写在这里
evaluate    反标准化后的 MAE / RMSE / MAPE / 按日 MAE
monitor     读训练日志并画网页
```

不新建 `data.py`、`sampling.py`、`loss.py`。`generator` 可以 import `retriever`；`retriever` 不许 import `generator`。

模块之间只传递下面三种对象，再加上检索器的 `WindowSet`。

### `Config`（`generator/config.py`）

只放生成器自己的字段：权重路径、检索开关、K、容量、训练和端口。日期、窗长、种子跟检索器走，从 checkpoint 的 `config` 读。命令行覆盖发生在 `train.main` 里，发生在任何模块被调用之前。

### `WindowSet`（`retriever/data.py`）

沿用检索器的四块数组。生成器的编码器只读查询的 `x` 和按行号查出的 `database.x` / `database.y`。当前查询的 `y` 只有 `train` 算损失、`evaluate` 算指标时才能读。

### `Retrieved`（`retrieve.py`）

每条查询对应 K 个检索库行号，形状 `(查询数, K)`。`K=0` 时第二维是 0。训练时先用行号取出 `Xi`、`Yi`，再送进生成器。

一次运行：

```text
train.main
  ├─ generator.Config()
  ├─ 读取 encoder.pt 里的 config → load_experiment（frozen / none 都走这里）
  ├─ frozen 时再构造 RetrieverEncoder，retrieve.topk(train / val / test)
  ├─ none 时 Retrieved 的 K=0
  ├─ ForecastGenerator
  └─ 每一轮
        ├─ 按行号取 Xi, Yi；只把 X, Xi, Yi 送进模型
        ├─ MSE(Ŷ_norm, Y_norm)
        ├─ evaluate 算验证 MAE，变低才覆盖 generator.pt
        └─ 结束后在测试集上报指标
```

每次运行单独存一份目录，里面有参数、数据切分摘要、逐步损失、验证 MAE 最好的模型。

## 9. 怎样算做完

1. 没有 `encoder.pt` 时拒绝开训。
2. `ForecastGenerator.forward` 的输入里没有当前查询的 `Y`。
3. 训练时网页上能看到训练 MSE 和验证 MAE 一起更新。
4. 测试给出总体 MAE、RMSE、MAPE，以及 7 天分日 MAE。
5. `retrieval_source=none` 能用同一套代码跑通，且切分与冻结检索一致。

## 10. 已确认的决定

下面几项已按你的确认固定。要改实验时，改 `generator/config.py` 里对应的值。

1. Stage B：冻结检索器，不联合训练。
2. 默认 `retrieval_source=frozen`，输入是 `X + (Xi, Yi)`。无检索用 `retrieval_source=none`，K=0。
3. 生成器是 token 拼接 Transformer，不用交叉注意力，也不用 Moment。
4. 标准化用当前 `X` 的均值和标准差，检索案例被放到当前负荷尺度上。
5. 默认 K=5。生成器容量 256 维、4 层、8 头，每 24 小时一个 patch。学习率 1e-4，最多 100 轮，按验证 MAE 存模型。
6. 数据切分完全跟检索器走：检索库在 2025-01-01 之前，训练用 2025，验证和测试从 2026 随机对半切。
7. 随机检索、过去相似 RAFT、Oracle 预测、`Yi-only` 都不在这一版。
