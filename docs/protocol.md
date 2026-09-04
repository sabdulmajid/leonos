# Leonos v1 protocol

Frozen before final-test inspection; exact accepted symbols and exchange dates are
added after the byte-level data audit.

## Question and adaptation

Leonos independently applies published components to a fixed U.S.-equity panel.
It is not an exact reproduction of Kronos's original markets or reported results.
The two substantive competitors are frozen `Kronos-base` and one pooled LightGBM;
their architectures and training histories differ, so a performance gap cannot
isolate pretraining as its sole cause.

Kronos's paper defines the investment score as the mean of forecast closes over
ten future sessions relative to the current close and uses 90 daily observations
for that experiment [R1]. Leonos applies exactly that score and target. The paper
reports that pretraining data extend through June 2024; this is an author-reported
cutoff, not an independently audited checkpoint guarantee. Paper inference settings
for price/investment are temperature 0.6, top-p 0.90, and ten sampled paths [R1].

## Information, target, and splits

For ticker `i` after exchange session `t`, the model receives only its last 90
completed OHLCV rows (including `t`) and known future calendar timestamps. It may
not receive ticker identity, cross-asset values, labels, news, fundamentals, or
future prices. Missing market candles are not imputed.

The realized label is:

```text
y[i,t] = mean(C[i,t+1], ..., C[i,t+10]) / C[i,t] - 1
```

This average-price appreciation label is not a directly executable ten-session
return. Prediction artifacts are saved before labels are joined. Development
labels end no later than 2023-12-31. Validation origins are exchange sessions in
July-December 2024 and labels end by 2024-12-31. After validation-only selection,
LightGBM is refit once on all labels ending by 2024-12-31. Final test origins begin
in 2025 and end early enough for ten observed future sessions. Kronos never fits.

## Data and adjustment gate

The only market download is the immutable `bars_1day` train/val/test snapshot in
`configs/base.yaml`. The dataset card's 51-equity universe, coverage, timezone, and
adjustment claims are hypotheses checked against downloaded bytes. Qlib is
evaluation software, not a guarantee that an imported dataset is unbiased.

Daily source timestamps are interpreted as session identifiers; a bar becomes
available only after its close. Leonos preserves source columns and will select one
split-consistent OHLCV basis only after auditing source collection code and known
corporate actions. Adjusted close is never mixed with incompatible raw OHLC. A
present-day historical snapshot is not called point-in-time-vintage data.

## Models

Kronos uses exact local snapshots of `NeoQuasar/Kronos-base` and
`NeoQuasar/Kronos-Tokenizer-base`, official code at the pinned revision, evaluation
mode, disabled gradients, 90 input sessions, and ten outputs. Missing amount follows
the inspected upstream predictor convention: it is estimated from volume times the
mean OHLC and is documented as an estimated turnover proxy, not observed turnover
or genuine VWAP. Ten decoded paths are averaged; that point path is not treated as
a calibrated predictive distribution. Logical batches and seeds are frozen in the
run manifest; changed batch shapes are not promised bitwise-identical.

LightGBM is pooled across historical ticker examples but each row contains only
causal OHLCV-derived features with effective lookback at most 90 sessions and known
calendar fields. This is called an OHLCV-supported Alpha158 adaptation. It overrides
Qlib's default label, omits genuine VWAP/amount factors, applies no cross-sectional
label normalization, and fits processors on development only. At most 12 declared
configurations use validation mean daily RankIC, early stopping, and deterministic
lower-complexity tie-breaking. All candidates remain in the search artifact.

## Signal evaluation

Primary RankIC is Spearman correlation **across equities for each origin date**,
using average ranks for ties—not correlation along one forecast path. Each test
date receives equal weight. Both models are restricted to their common predeclared
eligible observations. A constant score has undefined RankIC and is reported `NA`.

The primary statistic is mean daily `RankIC(Kronos) - RankIC(LightGBM)` with a
paired moving-block 95% bootstrap interval over complete date rows: block 20,
2,000 replicates, seed 42. Blocks 10 and 40 are declared sensitivities. Secondary
signal results are mean daily raw-score MAE (also basis points), zero-score MAE,
each mean RankIC, coverage, and runtime. Seed 42 is headline; seeds 43/44 show model
sampling/training sensitivity separately from market-date uncertainty.

## Portfolio

Both saved score series feed the same Qlib `TopkDropoutStrategy`: `topk=5`,
`n_drop=1`, `hold_thresh=5`, long-only, no leverage, 95% target exposure, USD 1m
illustrative capital, zero cash return. A signal formed after close `t` is indexed
by `t`; inspected Qlib code looks back one trading step and therefore trades it at
open `t+1` with no second shift. U.S. settings disable China-specific limits, lots,
taxes, benchmarks, and close fills. Primary proportional cost is 5 bps per side;
0 and 15 bps are sensitivities. Unfilled orders, valuation-only carries, fees, cash,
positions, and remaining unrealized holdings are retained. No forced final sale.

Net compounded wealth is reconciled to account value; Qlib `return` and `cost`
semantics are checked before use. Report net return, zero-cash Sharpe, maximum
drawdown, turnover, and actual costs, plus the 95%-invested equal-dollar buy-and-hold
reference established at the first eligible open.

## Sources

- [R1: Kronos paper v1](https://arxiv.org/html/2508.02739v1)
- [R2: official Kronos code](https://github.com/shiyu-coder/Kronos/tree/67b630e67f6a18c9e9be918d9b4337c960db1e9a)
- [R2: Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base/tree/2b554741eca47781b64468546e77fef3e85130e6)
- [R2: tokenizer](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base/tree/0e0117387f39004a9016484a186a908917e22426)
- [R3: pinned dataset](https://huggingface.co/datasets/twelvedata/financial-world-model/tree/88b972b547078237865255d9e15e4e16e1dd855f)
- [R3: collection source](https://github.com/twelvedata/twelvedata-world-model-dataset/tree/bbb4cbbfea92677e1a7cc0363be9a01c90e4fbc6)
- [R4/R6: Qlib](https://github.com/microsoft/qlib/tree/79633dd9506ea689e5400dea0197717b5b3d74b7)
- [R5: Qlib strategy docs](https://qlib.readthedocs.io/en/latest/component/strategy.html)
