# 检索器代码说明

这份说明只讲代码怎么接起来。实验规则仍以 `retriever/SPEC.md` 为准。

入口只有一个：`python -m retriever.train` 调用 `train.main`。其余文件不自己启动，只提供类和函数，由 `main` 按顺序调用。

## 一次运行怎么走

```text
train.main
  │
  ├─ Config()                          读默认参数，命令行可以覆盖其中几项
  ├─ pick_device()                     选择 cuda / mps / cpu
  │
  ├─ data.load_experiment(config)
  │     └─ 返回 4 个 WindowSet：检索库、训练、验证、测试，外加一份 summary
  │
  ├─ monitor.start_monitor(run_dir)    后台线程提供网页，网页只读 run_dir 里的文件
  │
  ├─ sampling.build_pools(train, database)
  ├─ sampling.build_pools(val, database)
  │     └─ 各返回一个 Pools，里面是检索库的下标，不是负荷本身
  │
  ├─ RetrieverEncoder(config)          唯一的神经网络
  │
  └─ 每一轮
        ├─ train.run_epoch(..., optimizer)     训练，会反传
        │     └─ train.batch_losses
        │           ├─ train.sample_candidates  从 Pools 抽下标
        │           ├─ train.encode             只把 X / Xi 送进编码器
        │           └─ loss.retriever_losses
        ├─ train.run_epoch(..., optimizer=None) 验证损失，不反传
        ├─ evaluate.evaluate_retrieval(..., val) 验证检索质量
        ├─ train.append_metric                  追加一行 metrics.jsonl，网页下一秒就能看到
        └─ train.save_model                     验证检索升高才覆盖 encoder.pt
  │
  └─ evaluate.evaluate_retrieval(model, database, test)
        ├─ evaluate.embed                再次调用 RetrieverEncoder，只吃 X
        └─ sampling.window_similarity    用 Yi 给检索结果打分，不进编码器
```

测试查询的 `WindowSet` 不进入 `build_pools`，也不进入 `run_epoch`。它只在训练结束后的 `evaluate_retrieval` 里出现。验证查询每轮也会走一次 `evaluate_retrieval`，用来存模型。

## 文件之间传递的两个对象

模块之间不互相读 CSV，也不互相写网页。它们只传递下面两种对象，再加上 `Config`。

### `Config`（`config.py`）

一个 dataclass，没有方法逻辑，只有字段和 `to_dict()`。窗长、日期、分位数、模型宽度、损失权重、端口都在这里。`data`、`sampling`、`model`、`loss`、`evaluate` 都只读它，不修改它。命令行覆盖发生在 `train.main` 里，发生在任何模块被调用之前。

### `WindowSet`（`data.py`）

一条时间窗口的集合。字段是四块对齐的数组：

| 字段 | 形状 | 谁可以读 |
| --- | --- | --- |
| `x` | `(案例数, 504)` | 编码器，以及 `sampling` 用来算过去相似度 |
| `y` | `(案例数, 168)` | 只有 `sampling` 和 `evaluate`。编码器的函数签名里没有它 |
| `origin_utc` | `(案例数,)` | 切分和打印日期 |
| `future_end_utc` | `(案例数,)` | 检查检索库有没有碰到查询 |

`subset(indices)` 用来从 2026 年查询里切出验证和测试。切开之后，`x[i]` 和 `y[i]` 仍是同一个预测起点。

### `Pools`（`sampling.py`）

一条查询对应三份检索库下标：`positive`、`hard`、`easy`。下标指向 `database` 这个 `WindowSet` 的行，不指向查询自己。训练时先用下标取出 `database.x`，再送进编码器。

## 每个文件做什么

### `data.py`：从 CSV 到 `WindowSet`

`load_experiment` 是对外的唯一入口，内部按这个顺序调用：

```text
load_hourly_load
  → pacific_midnight          把日期变成加州时间零点
  → _origins                  三次：检索库、2025 训练、2026 评估
        → _candidate_indices
              → _pacific_hour  每天只留本地 23:00
  → make_windows
        → _gather             按起点切出 X 和 Y
  → random_eval_split         只打乱 2026 年查询
  → WindowSet.subset
  → format_pacific            把时间写成给人看的字符串，放进 summary
```

泄漏检查写在 `load_experiment` 里：`database.future_end_utc.max()` 必须早于训练查询的 `origin_utc.min()`，否则直接抛出异常，`main` 不会往下训练。

### `sampling.py`：从 `Y` 得到下标，不创建模型

```text
build_pools(queries, database, config)
  ├─ window_similarity(queries.y, database.y)    未来相似度
  ├─ window_similarity(queries.x, database.x)    过去相似度，只为了找难负例
  │     └─ zscore_rows                           NumPy 版的按窗口标准化
  └─ classify_candidates                         按分位数交出三组下标
```

`pool_report` 只统计池子有多大，供 `main` 打印，不参与训练。

`evaluate.py` 也会调用 `window_similarity` 和 `classify_candidates`。评估和打标签用的是同一套规则，避免训练和测试各写一套阈值。

### `model.py`：只认识负荷窗口

`standardize_windows` 是 PyTorch 版的按窗口标准化，由 `RetrieverEncoder.forward` 调用。它和 `sampling.zscore_rows` 做同一件事，但一个在 GPU 张量上，一个在打标签的 NumPy 数组上，所以分成两个函数。

`RetrieverEncoder.forward(windows)` 的输入形状是 `(batch, 504)` 的原始负荷。它自己做四步：按窗口标准化，`patch` 把每 24 小时变成一个 token，`encoder` 做 Transformer，`output` 把 token 平均后投影成单位向量。返回值形状是 `(batch, d_model)`，默认 128 维。

这个类不读取 `Config` 以外的任何实验状态，也不知道检索库的存在。

### `loss.py`：只认识向量

`retriever_losses(query, positive, negative, temperature, margin)` 收到的已经是单位向量。`cosine` 就是点积。它返回两个标量：对比损失和排序损失。加权相加发生在 `train.batch_losses`，因为权重属于训练配置，不属于损失公式本身。

### `train.py`：把一次 batch 拼出来

`run_epoch` 负责打乱查询下标、按 `batch_size` 切片。`optimizer` 不是 `None` 时才反传；验证把 `optimizer` 传成 `None`，并包在 `torch.no_grad()` 里。

一个 batch 在 `batch_losses` 里的接法：

```text
sample_candidates(pools, 查询行号)
  → 得到 database 的行号
database.x[那些行号]          只取 X，不取 Y
encode
  → RetrieverEncoder.forward（内部标准化）
retriever_losses
```

`encode` 是训练和模型之间的窄接口：外面放进去的是 NumPy 负荷，出来的是单位向量。

`append_metric` 每写一行就关闭文件。`monitor.py` 不接收损失数值，它只是反复读取这个文件。两边只通过 `metrics.jsonl` 通信，所以网页挂了也不会打断反传。

`save_model` 把 `state_dict` 和 `config.to_dict()` 一起写入。评估前 `main` 会从 `encoder.pt` 重新 `load_state_dict`，用的是验证检索未来相似度最高的那一轮，不是最后一轮。

### `evaluate.py`：检索，然后查表

```text
embed(model, test.x)          查询向量
embed(model, database.x)      检索库向量
点积，取前 5 个 database 行号
database.y[那些行号]          这就是按行号查出 Yi
window_similarity             拿 Yi 和测试查询的 Y 比较
```

`embed` 内部调用的是 `RetrieverEncoder`，与训练共用同一个前向过程。Oracle 和随机对照不调用编码器，只比较 `window_similarity` 的分数。

### `monitor.py` 和 `monitor.html`

`start_monitor` 在后台线程启动 `ThreadingHTTPServer`。请求进来时：

| 路径 | 读什么 |
| --- | --- |
| `/` | `monitor.html` |
| `/api/summary` | `summary.json`，由 `load_experiment` 的返回值写成 |
| `/api/metrics` | `read_jsonl(metrics.jsonl)` |

`monitor.html` 没有 Python 类。页面每秒请求上面两个接口，自己画曲线。训练循环不 import 网页。

## 谁不能看见 `Y`

| 函数 | 输入里有没有当前查询的 `Y` |
| --- | --- |
| `RetrieverEncoder.forward` | 没有 |
| `train.encode` | 没有，调用处只传入 `.x` |
| `loss.retriever_losses` | 没有，只收向量 |
| `sampling.build_pools` | 有，用来生成下标 |
| `evaluate.evaluate_retrieval` | 有，用来给已经检索到的案例打分 |

历史库的 `Yi` 同样不进编码器。评估时它是 `database.y[检索到的行号]`，发生在前向计算之后。
