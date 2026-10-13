# Future-Aware RAFT for Load Forecasting — Development Specification

## 0. Document Status

- **Project type:** Research prototype / thesis experiment
- **Primary objective:** Develop and evaluate a future-aware retrieval-augmented load forecasting framework based on the RAFT paradigm.
- **Core hypothesis:** Historical examples should be retrieved according to their usefulness for forecasting the future, rather than only according to similarity between observed past load patterns.
- **Current implementation target:** Deterministic point load forecasting first; probabilistic forecasting can be added after the retrieval pipeline is validated.
- **Important constraint:** The true future target `Y` must never be provided to the retriever at inference time. `Y` and candidate future trajectories `Yi` may only be used to construct training supervision and to train/evaluate the forecasting generator.

---

# 1. Research Objective

## 1.1 Problem Definition

Given an observed historical window:

\[
X = [x_{t-L+1}, ..., x_t]
\]

predict the future load:

\[
Y = [x_{t+1}, ..., x_{t+H}]
\]

where:

- `L` = input/history length
- `H` = forecast horizon

The proposed framework retrieves historical forecasting cases:

\[
(X_i, Y_i)
\]

from a historical database and uses them as additional evidence for forecasting the current target:

\[
\hat{Y} = G(X, \{X_i,Y_i\}_{i=1}^{K})
\]

where:

- `X` = current observed context
- `Xi` = historical candidate context
- `Yi` = known future trajectory following `Xi`
- `K` = number of retrieved cases
- `G` = forecasting generator/model

---

# 2. Research Hypothesis

## H1 — Retrieval Benefit

Using historical retrieved cases should improve forecasting performance compared with a forecasting model that receives only the current context.

## H2 — Conventional Similarity Limitation

Retrieving cases solely according to similarity of observed past patterns does not necessarily retrieve the most useful forecasting examples.

## H3 — Future-Aware Retrieval

A retriever trained with future-trajectory information as supervision can learn to identify historical cases that are more useful for forecasting.

## H4 — Different Past, Similar Future

Historical cases with substantially different past patterns but similar future trajectories can still be useful retrieval examples.

## H5 — Hard Negatives

Historical cases with highly similar past patterns but substantially different future trajectories are particularly useful hard negatives because they expose the weakness of naive past-similarity retrieval.

---

# 3. Overall System

The complete proposed system is:

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

Training additionally uses the known current future `Y`:

```text
Current:
X ─────────→ Y

Candidate:
Xi ────────→ Yi

X, Xi ─────────→ Retriever score
Y, Yi ─────────→ Future relevance supervision
```

The future target `Y` is never an inference-time retriever input.

---

# 4. Data Representation

## 4.1 Forecasting Sample

Each sample must contain:

```text
sample_id
timestamp
X
Y
optional exogenous features
```

Conceptually:

```text
X:
[t-L+1, ..., t]

Y:
[t+1, ..., t+H]
```

## 4.2 Recommended Initial Configuration

Start with:

- History length: `168 hours`
- Forecast horizon: `24 hours`
- Sampling frequency: `hourly`

This corresponds to:

```text
7 days observed
→
24 hours forecast
```

These values must remain configurable.

Example configuration:

```yaml
history_length: 168
forecast_horizon: 24
frequency: 1H
```

Do not hard-code these values inside model code.

---

# 5. Input Features

## 5.1 Required Baseline Feature

At minimum:

```text
load
```

## 5.2 Optional Exogenous Features

The framework should support:

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

These should be configurable rather than permanently coupled to the model.

## 5.3 Feature Groups

Maintain separate feature groups:

```text
load_features
calendar_features
weather_features
other_exogenous_features
```

This allows ablation experiments.

---

# 6. Dataset Construction

## 6.1 Sliding Window

For each timestamp `t`:

```text
Past:
[t-L+1, ..., t]

Future:
[t+1, ..., t+H]
```

Create one forecasting sample.

Example:

```text
2025-01-08 00:00 → 2025-01-14 23:00
                    ↓
2025-01-15 00:00 → 2025-01-15 23:00
```

## 6.2 Historical Candidate Definition

Every eligible historical timestamp `ti` generates:

```text
Xi = data[ti-L+1 : ti]
Yi = data[ti+1 : ti+H]
```

The pair:

```text
(Xi, Yi)
```

is one retrievable historical forecasting case.

## 6.3 Temporal Leakage Prevention

Candidate cases must satisfy:

```text
candidate_future_end < current_sample_start
```

for strict historical retrieval.

Never allow:

```text candidate X or Y
```

to contain information from the current target period.

For a current target beginning at `T`, candidate data must end before `T`.

---

# 7. Dataset Split

Use chronological splitting.

Recommended:

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

Do NOT randomly shuffle timestamps across train/validation/test.

Example:

```yaml
train: 70%
validation: 15%
test: 15%
split_method: chronological
```

Exact percentages are configurable.

## 7.1 Retrieval Database

The retrieval database must be defined separately for each experiment.

Default:

```text
Training period → retrieval database for training
Training + validation period → retrieval database for validation/test only if explicitly allowed by the experimental protocol
```

For the cleanest evaluation of generalization, the default test protocol should retrieve only from historical data that existed before each test query.

---

# 8. Normalization

Load values can differ substantially across periods.

The implementation must support:

### Option A — Global normalization

\[
x' = \frac{x-\mu_{train}}{\sigma_{train}}
\]

Use statistics calculated from training data only.

### Option B — Window normalization

For each window:

\[
x'_t = \frac{x_t-\mu_X}{\sigma_X}
\]

This emphasizes shape similarity.

### Recommended experiments

Compare:

1. Global normalization
2. Per-window z-score
3. Raw load + scale information

Do not silently mix normalization strategies.

---

# 9. Retrieval Unit

The fundamental retrieval object is:

```text
HistoricalCase {
    case_id
    end_timestamp
    X
    Y
    metadata
}
```

Example:

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

# 10. Retriever Architecture

## 10.1 Retriever Input

The retriever must receive:

```text
current X
candidate Xi
```

or their encoded representations.

At inference:

```text
Retriever(X)
```

is used to retrieve candidate cases.

The retriever must NOT receive current `Y`.

## 10.2 Encoder

Initial implementation:

```text
Time-Series Transformer Encoder
```

Architecture should be configurable.

Example:

```yaml
retriever:
  encoder: transformer
  d_model: 256
  num_layers: 4
  num_heads: 8
  dropout: 0.1
```

These are starting values, not fixed research conclusions.

## 10.3 Embedding

The encoder converts each time-series window into:

\[
z_X = Encoder(X)
\]

and:

\[
z_i = Encoder(X_i)
\]

Similarity:

\[
S_p(X,X_i)=cos(z_X,z_i)
\]

Initial implementation should use cosine similarity.

Alternative similarity functions may be evaluated later.

---

# 11. Candidate Retrieval

## 11.1 Retrieval Database

Each historical candidate must have:

```text
case_id
embedding
X
Y
timestamp
metadata
```

## 11.2 ANN Index

For scalability, use an ANN vector index.

Possible implementations:

```text
FAISS
```

Initial retrieval:

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

Where:

- `M > K`
- `M` allows temporal/duplicate filtering before final selection.

Example:

```yaml
retrieval:
  candidate_pool_size: 100
  top_k: 5
```

---

# 12. Candidate Filtering

Before accepting a candidate:

1. Candidate must be chronologically valid.
2. Candidate must not overlap the current target.
3. Candidate must satisfy minimum temporal separation if configured.
4. Duplicate or near-duplicate windows should be removed.
5. Missing-data constraints must be satisfied.

Optional:

```text same timestamp-of-day
same weekday
same season
same holiday status
```

These should be experimental filters, not mandatory assumptions.

---

# 13. Future-Aware Supervision

For each current training sample:

```text Current:
X → Y

Candidate:
Xi → Yi
```

Calculate:

### Past similarity

\[
S_p = sim(X,X_i)
\]

### Future similarity

\[
S_f = sim(Y,Y_i)
\]

The exact similarity function must be configurable.

Initial recommendation:

```text normalized MSE
+
correlation
```

For example:

\[
D_f(Y,Y_i)=MSE(normalize(Y),normalize(Y_i))
\]

and convert distance to relevance:

\[
S_f = \exp(-D_f/\tau)
\]

where `tau` is a temperature parameter.

---

# 14. Positive / Negative Taxonomy

Use a two-dimensional classification.

| Past Similarity | Future Similarity | Category |
|---|---|---|
| High | High | Strong Positive |
| High | Low | Hard Negative |
| Low | High | Future Positive |
| Low | Low | Easy Negative |

Important:

### Strong Positive

Historical case has similar past and useful future.

### Hard Negative

Historical case looks similar from the observed past but has a dissimilar future.

This is especially important.

### Future Positive

Historical case has a different past but a similar future.

This should NOT automatically be treated as negative.

### Easy Negative

Historical case has both different past and different future.

---

# 15. Thresholds

Do not hard-code arbitrary thresholds without validation.

Initial implementation can use quantiles.

For example:

```text high similarity = top 20%
low similarity = bottom 50%
```

or:

```yaml
positive_future_quantile: 0.8
negative_future_quantile: 0.2
```

Thresholds must be determined using the training set only.

Perform sensitivity analysis later.

---

# 16. Retriever Training Objective

There are two recommended stages.

## Stage 1 — Simple Contrastive Baseline

Use:

```text
X
positive Xi
negative Xi
```

and an InfoNCE-style loss:

\[
L_{retriever}
=
-\log
\frac{
\exp(sim(z_X,z_{pos})/\tau)
}{
\exp(sim(z_X,z_{pos})/\tau)
+
\sum_j \exp(sim(z_X,z_{neg,j})/\tau)
}
\]

This provides a clear baseline.

## Stage 2 — Future-Aware Ranking

Use future similarity as a relevance signal.

Desired behavior:

\[
R(X,X_a)>R(X,X_b)
\]

if:

\[
S_f(Y,Y_a)>S_f(Y,Y_b)
\]

A ranking loss can be used:

\[
L_{rank}
=
\max(0,m-R(X,X^+)+R(X,X^-))
\]

where:

- `m` = margin
- `X+` = more useful historical candidate
- `X-` = less useful candidate

---

# 17. Recommended Retriever Loss

A combined objective can be implemented:

\[
L_R =
\lambda_{con}L_{contrastive}
+
\lambda_{rank}L_{ranking}
\]

Start with:

```yaml
lambda_contrastive: 1.0
lambda_ranking: 1.0
```

and perform ablations.

Do not assume this combined loss is optimal.

---

# 18. Generator

## 18.1 Generator Input

The generator receives:

```text
Current X
+
Retrieved historical cases
```

Initial proposed representation:

\[
G(X,\{X_i,Y_i\}_{i=1}^{K})
\rightarrow \hat{Y}
\]

Therefore each retrieved example contains:

```text
Xi
Yi
```

## 18.2 Why Include Xi?

`Xi` provides historical context explaining the conditions under which `Yi` occurred.

However, this must be tested.

Ablation:

```text
A: Current X only
B: Current X + retrieved Yi
C: Current X + retrieved Xi + Yi
```

---

# 19. Generator Architecture

Initial model:

```text
Transformer-based forecasting model
```

Possible structure:

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

The exact architecture should remain modular.

---

# 20. Generator Loss

For deterministic forecasting:

### MAE

\[
L_{MAE}
=
\frac{1}{H}
\sum_{h=1}^{H}|Y_h-\hat{Y}_h|
\]

### MSE

\[
L_{MSE}
=
\frac{1}{H}
\sum_{h=1}^{H}(Y_h-\hat{Y}_h)^2
\]

Initial training can use MSE.

Evaluate with:

```text MAE
RMSE
MAPE
```

Do not optimize directly for MAPE if zero/near-zero loads are possible.

---

# 21. End-to-End Training Strategy

Use staged training first.

## Stage A — Train Retriever

Input:

```text X, Xi
```

Supervision:

```text Y, Yi
```

Output:

```text retrieval relevance
```

## Stage B — Freeze Retriever

Retrieve top-K historical examples.

Train generator:

```text X + retrieved Xi,Yi → Y
```

## Stage C — Optional Joint Fine-Tuning

After the pipeline is stable:

```text Retriever
    ↓
Top-K
    ↓
Generator
```

Joint optimization may be attempted.

However, initial experiments should avoid joint training because retrieval changes can make the system unstable and make debugging difficult.

---

# 22. Preventing Data Leakage

This section is mandatory.

## Retriever

At inference, retriever receives only information available at prediction time.

Allowed:

```text past load
known calendar features
known weather information
```

Not allowed:

```text actual future load Y
future load-derived statistics
future target-derived labels
```

## Generator

Generator receives retrieved `Yi` because `Yi` belongs to historical cases.

Generator must never receive the current target `Y`.

## Training Labels

`Y` can be used to:

```text calculate future similarity
construct positive/negative labels
calculate forecasting loss
```

This is allowed because it is training supervision.

---

# 23. Important Leakage Edge Case

If a candidate's future `Yi` overlaps the current sample's target period, reject the candidate.

For current target:

```text T+1 ... T+H
```

candidate must satisfy:

```text candidate_future_end < T+1
```

This must be explicitly checked in code.

---

# 24. Baseline Experiments

At minimum implement:

## Baseline 0 — No Retrieval

```text
X → Forecasting Model → Y
```

## Baseline 1 — Random Retrieval

```text
X
+
random historical Xi,Yi
→ Generator
```

Tests whether retrieval itself is useful.

## Baseline 2 — Past-Similarity Retrieval

```text
X
↓
similarity(X,Xi)
↓
Top-K
↓
Generator
```

This is the conventional RAFT-style baseline.

## Baseline 3 — Oracle Future Retrieval

Training/evaluation diagnostic only:

```text
Y
↓
similarity(Y,Yi)
↓
Top-K
```

This must NOT be presented as a deployable inference method.

Its purpose is to estimate the upper bound of future-aware retrieval quality.

## Proposed — Learned Future-Aware Retriever

```text
X
↓
learned Retriever
↓
Top-K useful historical cases
↓
Generator
```

Training uses future-aware supervision.

---

# 25. Recommended Ablation Studies

## Retrieval Information

```text A: X only
 B: X + calendar
 C: X + weather
 D: X + calendar + weather
```

## Retrieved Content

```text A: Yi only
B: Xi + Yi
```

## Retrieval Strategy

```text A: Random
B: Past similarity
C: Learned retriever
D: Future-aware learned retriever
```

## Number of Retrieved Cases

```text K = 1
K = 3
K = 5
K = 10
K = 20
```

## Hard Negative Strategy

```text A: Random negatives
B: ANN hard negatives
C: Past-similar / future-dissimilar negatives
D: Mixed negatives
```

---

# 26. Main Experiment Table

The main result table should eventually resemble:

| Method | Retrieval | Future-aware Training | MAE | RMSE | MAPE |
|---|---|---|---:|---:|---:|
| Forecasting Baseline | None | No | | | |
| Random Retrieval | Random | No | | | |
| RAFT-style | Past similarity | No | | | |
| Learned Retriever | Learned | No | | | |
| Proposed | Future-aware | Yes | | | |

---

# 27. Retrieval Evaluation

Do not evaluate only final forecasting metrics.

Evaluate retrieval itself.

## Recall@K

If ground-truth relevant historical cases are defined:

\[
Recall@K =
\frac{\# relevant\ cases\ retrieved\ in\ top-K}
{\# relevant\ cases}
\]

## Future Similarity of Retrieved Cases

Report:

```text average Future Similarity @ K
```

Compare:

```text random
past similarity
proposed retriever
```

## Hard Negative Retrieval Rate

Measure how often the retriever selects:

```text high Past Similarity
+
low Future Similarity
```

A successful retriever should reduce these cases.

---

# 28. Forecasting Evaluation by Horizon

Do not only report one aggregate number.

Report:

```text
1–6 hours
7–12 hours
13–18 hours
19–24 hours
```

or another appropriate horizon grouping.

Reason:

Retrieval may help long-horizon forecasting differently from short-horizon forecasting.

---

# 29. Forecasting Evaluation by Regime

If data allows, evaluate separately for:

```text weekday
weekend
holiday
summer
winter
high-load periods
low-load periods
extreme weather
```

This can reveal whether retrieval is especially useful during unusual regimes.

---

# 30. Visualization Requirements

At minimum produce:

## 30.1 Forecast Plot

```text
Actual Y
Predicted Ŷ
```

## 30.2 Retrieved Cases

Plot:

```text
Current X
Retrieved X1, X2, ..., XK
```

and separately:

```text
Actual Y
Retrieved Y1, Y2, ..., YK
Predicted Ŷ
```

## 30.3 Retrieval 2D Analysis

Plot candidates using:

```text x-axis = Past Similarity
y-axis = Future Similarity
```

Color/label:

```text Strong Positive
Future Positive
Hard Negative
Easy Negative
```

This visualization is especially important for validating the proposed research idea.

---

# 31. Data Pipeline

Recommended pipeline:

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

Each stage should save reproducible intermediate metadata.

---

# 32. Missing Data

Define explicit rules.

For example:

```yaml
max_missing_ratio: 0.05
```

If missing ratio exceeds threshold:

```text discard window
```

Otherwise:

```text interpolate / impute
```

The imputation strategy must be documented.

Never allow future information to enter past-window imputation.

---

# 33. Reproducibility

Every experiment must save:

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

Recommended:

```yaml
seed: 42
```

Use multiple seeds for final experiments.

---

# 34. Suggested Project Structure

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

# 35. Development Phases

## Phase 1 — Data Pipeline

Implement:

- loading raw load data
- cleaning
- resampling
- chronological split
- sliding-window generation
- normalization
- historical case construction

**Deliverable:** reproducible `(X,Y)` dataset.

---

## Phase 2 — Forecasting Baseline

Implement:

```text
X → Transformer → Y
```

No retrieval.

**Deliverable:** baseline MAE/RMSE/MAPE.

---

## Phase 3 — Conventional RAFT Baseline

Implement:

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

**Deliverable:** comparison against no-retrieval baseline.

---

## Phase 4 — Retriever

Implement:

```text
Transformer encoder
+
cosine similarity
+
ANN retrieval
```

**Deliverable:** learned retrieval system.

---

## Phase 5 — Future-Aware Supervision

Implement:

```text
future similarity
+
positive/negative construction
+
hard-negative mining
+
contrastive/ranking loss
```

**Deliverable:** future-aware retriever.

---

## Phase 6 — Generator Integration

Implement:

```text
Current X
+
Retrieved Xi,Yi
→
Generator
→
Ŷ
```

**Deliverable:** complete proposed framework.

---

## Phase 7 — Ablation

Run:

- no retrieval
- random retrieval
- RAFT-style retrieval
- learned retrieval
- future-aware retrieval
- different K
- different feature groups
- Yi-only vs Xi+Yi
- different negative strategies

---

## Phase 8 — Analysis

Analyze:

- retrieval quality
- forecasting accuracy
- hard negatives
- future positives
- different load regimes
- forecast horizons
- failure cases

---

# 36. Critical Failure Cases to Investigate

The system must log examples where:

### Case A

```text
Past very similar
Future very different
```

Potential hard negative.

### Case B

```text
Past very different
Future very similar
```

Potential future positive.

### Case C

```text
Retriever selects an apparently similar case
but generator becomes worse
```

### Case D

```text
Retrieved cases are nearly duplicates
```

### Case E

```text
Retrieved case comes from an inappropriate season/regime
```

These examples should be manually inspected.

---

# 37. Important Research Interpretation

The proposed model should NOT claim:

> "The model retrieves the most similar historical load."

The stronger and more accurate claim is:

> **The model learns to retrieve historical forecasting cases that are expected to be useful for future load prediction.**

This distinction should be maintained throughout the paper.

---

# 38. Recommended Terminology

Use:

### Historical Case

\[
(X_i,Y_i)
\]

rather than simply "document".

### Query / Current Case

\[
X
\]

### Past Similarity

\[
S_p
\]

### Future Similarity

\[
S_f
\]

### Future-aware Retrieval

Training retrieval using future trajectory information as supervision.

### Hard Negative

High past similarity but low future similarity.

### Future Positive

Low past similarity but high future similarity.

### Retrieved Future

`Yi` associated with a retrieved historical case.

---

# 39. Final Mathematical Definition

Current forecasting task:

\[
Y=f(X)
\]

Standard forecasting:

\[
\hat{Y}=G(X)
\]

RAFT-style forecasting:

\[
\hat{Y}
=
G(X,\mathcal{R}(X))
\]

where:

\[
\mathcal{R}(X)
=
\{(X_i,Y_i)\}_{i=1}^{K}
\]

The proposed future-aware retriever is trained using:

\[
R_\theta(X,X_i)
\]

while future trajectories provide supervision:

\[
S_f(Y,Y_i)
\]

The desired relationship is:

\[
R_\theta(X,X_i)
\approx
f(S_f(Y,Y_i),S_p(X,X_i))
\]

but at inference:

\[
\boxed{
\mathcal{R}(X)
\text{ uses only information available before the forecast horizon.}
}
\]

---

# 40. Final Research Pipeline

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

# 41. Definition of Success

The project is successful if experiments demonstrate:

1. Retrieval improves forecasting compared with the no-retrieval baseline.
2. Conventional past-similarity retrieval can select hard negatives.
3. Future-positive cases exist in the dataset and can be useful.
4. The proposed future-aware retriever retrieves cases with higher future relevance.
5. Future-aware retrieval improves forecasting metrics over conventional RAFT-style retrieval.
6. The improvement remains under chronological, leakage-free evaluation.
7. The improvement can be explained through retrieval-quality analysis rather than only aggregate forecasting metrics.

---

# 42. First Implementation Priority

Do NOT implement the entire system at once.

Implement in this exact order:

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

The first milestone should be:

> **A working RAFT-style baseline on the load dataset before introducing future-aware retriever training.**

This guarantees that any improvement from the proposed method can be attributed to the retrieval-learning component rather than an unverified end-to-end pipeline.
