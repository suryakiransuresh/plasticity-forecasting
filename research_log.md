Experiment 0.1–0.2:
A single synthetic factual update ("Velora → Nymara") improved all five
pretrained protected probes rather than damaging them.

Result:
Acquisition gain = +4.1900 loss reduction
Mean protected loss change = -0.5212

Interpretation:
Pretrained factual probes are insufficiently controlled for studying
interference. Next experiment will establish synthetic protected knowledge
before introducing new knowledge.


# Research Log — Selective Test-Time Plasticity

## Experiment 0 — Controlled Interference Validation

**Date:** 2026-08-25  
**Model:** GPT-2 Small (124M)  
**Device:** Apple MPS  
**Seed:** 42

### Objective

Verify that learning a new synthetic fact can produce measurable interference with previously learned controlled memories.

### Controlled Memory Setup

Five synthetic factual associations were first learned:

- Velora → Nymara
- Torvane → Eloria
- Caldris → Vensar
- Merovia → Talune
- Zoravia → Pelith

After 20 epochs of controlled training, answer-only losses were:

| Memory | Loss |
|---|---:|
| Velora → Nymara | 0.0172 |
| Torvane → Eloria | 0.0161 |
| Caldris → Vensar | 0.0190 |
| Merovia → Talune | 0.0117 |
| Zoravia → Pelith | 0.0025 |

This trained state was saved as the `controlled_memory` checkpoint.

### New Experience

New synthetic association:

**Ravelle → Sorin**

Before adaptation:

- New-fact loss: 7.6220

Adaptation configuration:

- Full-model update
- AdamW
- Learning rate: 1e-5
- Weight decay: 0
- 10 update steps
- Training loss applied only to answer tokens

After adaptation:

- New-fact loss: 1.5347
- Acquisition gain: **6.0873**

### Protected-Memory Damage

| Protected Memory | Before | After | Damage |
|---|---:|---:|---:|
| Velora → Nymara | 0.0172 | 0.1693 | +0.1520 |
| Torvane → Eloria | 0.0161 | 0.1421 | +0.1259 |
| Caldris → Vensar | 0.0190 | 0.0925 | +0.0734 |
| Merovia → Talune | 0.0117 | 0.0699 | +0.0582 |
| Zoravia → Pelith | 0.0025 | 0.0320 | +0.0295 |

**Mean functional damage: +0.0878**

### Observation

The model strongly acquired the new association while all five previously learned synthetic memories showed increased loss.

Damage was heterogeneous across memories rather than uniform.

This establishes that the controlled experimental setup can produce measurable acquisition–interference behavior.

### Interpretation

This result does **not** yet validate the main research hypothesis.

Only one update strategy was tested. The central hypothesis requires demonstrating that different candidate updates, starting from the same checkpoint and learning the same experience, produce meaningfully different acquisition–damage trade-offs.

### Next Experiment

Compare multiple candidate updates from the identical `controlled_memory` checkpoint.

Initial candidate families:

1. Full model — LR 1e-6
2. Full model — LR 1e-5
3. Full model — LR 5e-5
4. Early-layer update
5. Middle-layer update
6. Late-layer update

For every candidate, measure:

- acquisition gain;
- mean protected-memory damage;
- individual memory damage.

Key question:

**Can two candidate updates achieve similar acquisition while producing substantially different functional damage?**

## Experiment 0.4 — Candidate Update Comparison

**Date:** 2026-08-25  
**Model:** GPT-2 Small (124M)  
**Starting checkpoint:** `controlled_memory`  
**Device:** Apple MPS  
**Seed:** 42

### Objective

Test whether different candidate update strategies, starting from the exact same controlled-memory checkpoint and learning the exact same new fact, produce different acquisition–damage trade-offs.

### New Experience

New synthetic association:

**Ravelle → Sorin**

All candidates started from the same saved controlled-memory checkpoint.

Each candidate was trained for 10 steps using answer-only loss and AdamW with zero weight decay.

### Candidate Strategies

1. Full-model update — learning rate `1e-6`
2. Full-model update — learning rate `1e-5`
3. Full-model update — learning rate `5e-5`
4. Early transformer layers only — learning rate `1e-5`
5. Middle transformer layers only — learning rate `1e-5`
6. Late transformer layers only — learning rate `1e-5`

For GPT-2 Small:

- Early layers: 0–3
- Middle layers: 4–7
- Late layers: 8–11

### Results

| Candidate | Acquisition Gain | Mean Damage |
|---|---:|---:|
| Full model, LR 1e-6 | 1.2863 | 0.0040 |
| Full model, LR 1e-5 | 5.9475 | 0.0938 |
| Full model, LR 5e-5 | 7.6220 | 4.6729 |
| Early layers | 2.9524 | 0.0154 |
| Middle layers | 3.3353 | 0.0238 |
| Late layers | 3.5248 | 0.0138 |

### Individual Protected-Memory Damage

| Protected Memory | Full 1e-6 | Full 1e-5 | Full 5e-5 | Early | Middle | Late |
|---|---:|---:|---:|---:|---:|---:|
| Velora → Nymara | +0.0082 | +0.1656 | +4.1750 | +0.0304 | +0.0494 | +0.0196 |
| Torvane → Eloria | +0.0031 | +0.1468 | +5.6981 | +0.0128 | +0.0207 | +0.0203 |
| Caldris → Vensar | +0.0038 | +0.0935 | +2.4318 | +0.0152 | +0.0283 | +0.0156 |
| Merovia → Talune | +0.0040 | +0.0471 | +4.4622 | +0.0156 | +0.0160 | +0.0094 |
| Zoravia → Pelith | +0.0007 | +0.0161 | +6.5974 | +0.0028 | +0.0044 | +0.0041 |

### Key Observation

Candidate update choice substantially affected both acquisition and protected-memory damage.

The most aggressive full-model update (`5e-5`) achieved the highest acquisition but caused severe interference with previously learned memories.

More importantly, differences were also observed among candidates using the same learning rate.

The late-layer candidate achieved:

- higher acquisition than the middle-layer candidate;
- lower mean damage than the middle-layer candidate.

Therefore the middle-layer candidate was dominated by the late-layer candidate for this episode.

### Interpretation

This is the first result supporting the core premise that alternative updates for the same learning experience can produce meaningfully different acquisition–damage outcomes.

However, the result is based on only:

- one new learning episode;
- one model checkpoint;
- one synthetic task family;
- one random seed.

It is therefore insufficient to claim that update consequences are systematically structured or predictable.

The learning-rate comparison is also partly expected: more aggressive full-model updates produced both greater acquisition and greater damage. The more scientifically interesting result is the difference between layer-specific updates under the same learning rate.

### Current Status

**Feasibility check passed provisionally.**

There is sufficient candidate-level variation to justify expanding the experiment.

Next Experiment

Run the same six candidate strategies across multiple independent novel associations.

Initial target:

20 novel learning episodes × 6 candidates = 120 candidate outcomes

The next questions are:

Does the gain–damage trade-off persist across episodes?

Does the best candidate change depending on the new experience?

Is one static candidate consistently optimal?

If the best candidate changes, can pre-update properties predict which candidate will be preferable?

A changing optimal candidate would provide stronger motivation for counterfactual update forecasting.


## Experiment 0.5 — Seed-Controlled Multi-Episode Candidate Sweep

**Date:** 2026-08-25  
**Model:** GPT-2 Small (124M)  
**Starting checkpoint:** `controlled_memory`  
**Device:** Apple MPS  
**Base seed:** 42  
**Episodes:** 20 synthetic factual associations  
**Candidates per episode:** 6  
**Total candidate outcomes:** 120

### Objective

Test whether the gain–damage differences observed in the single-episode candidate comparison persist across multiple new experiences.

A second objective was to remove a stochastic confound from the first sweep by ensuring that all six candidate strategies within the same episode use the same dropout seed.

### Experimental Design

Each episode introduced one novel synthetic association of the form:

**The capital of X is Y**

All candidate updates began from the exact same `controlled_memory` checkpoint containing five protected synthetic associations.

Candidate strategies:

1. Full model — LR `1e-6`
2. Full model — LR `1e-5`
3. Full model — LR `5e-5`
4. Early transformer layers only — LR `1e-5`
5. Middle transformer layers only — LR `1e-5`
6. Late transformer layers only — LR `1e-5`

All candidates were trained for 10 steps using answer-only loss and zero weight decay.

The same random seed was used across all six candidates within each episode.

### Aggregate Results

| Candidate | Mean Acquisition Gain | Mean Damage | Mean Update Norm |
|---|---:|---:|---:|
| Full model, LR 1e-6 | 1.592 | 0.0022 | 0.054 |
| Early layers | 3.903 | 0.0092 | 0.271 |
| Middle layers | 4.177 | 0.0163 | 0.297 |
| Late layers | 5.237 | 0.0117 | 0.322 |
| Full model, LR 1e-5 | 7.276 | 0.1958 | 0.517 |
| Full model, LR 5e-5 | 8.403 | 4.6896 | 2.218 |

### Main Observations

Candidate choice continued to produce substantially different acquisition–damage outcomes after controlling for stochastic seed differences.

The most aggressive full-model update (`5e-5`) achieved high acquisition but caused severe damage to previously learned memories.

The late-layer candidate consistently produced stronger acquisition than the other layer-specific candidates:

- Late > early gain in 20/20 episodes
- Late > middle gain in 19/20 episodes

Late-layer updating also dominated middle-layer updating, meaning higher acquisition and lower damage simultaneously, in 16/20 episodes.

This confirms that update location affects the gain–damage trade-off even when the learning rate, number of training steps, and stochastic seed are controlled.

### Update Magnitude Analysis

Across all candidate types, update norm was strongly associated with damage.

However, this relationship became weak when restricting the analysis to the early-, middle-, and late-layer candidates.

For the layer-specific candidates:

- update norms were relatively similar;
- damage still differed meaningfully;
- update magnitude alone did not explain the observed interference.

This suggests that **where and/or in what direction the model is updated may matter beyond how much the parameters move**.

### Important Failure Mode / Confound Discovered

Several high-damage episodes contained lexical similarity between the newly learned answer and one of the protected answers.

Examples include:

- New: `Pelor` vs protected: `Pelith`
- New: `Talen` vs protected: `Talune`
- New: `Elsin` vs protected: `Eloria`

These episodes showed unusually large damage to the lexically related protected memory.

A preliminary string-similarity sanity check indicated a moderate relationship between answer similarity and maximum protected-memory damage.

This raises the possibility that some observed interference is driven by token-level or lexical competition rather than a more general form of functional interference.

This must be controlled in future experiments.

### Pareto / Static Strategy Observation

The late-layer candidate appeared on the gain–damage Pareto frontier across all 20 episodes.

This creates an important limitation for the current task family.

If late-layer updating is consistently a strong static strategy for synthetic factual associations, a learned counterfactual predictor may not be necessary.

The core research hypothesis requires showing that **different experiences favor different update strategies**.

### Interpretation

Experiment 0.5 provides stronger evidence that candidate updates can produce systematic differences in acquisition and damage.

However, the experiment is still limited to a single task family:

**synthetic capital associations**

Therefore it does not yet establish experience-dependent plasticity.

The experiment also revealed a potential lexical/token-overlap confound that must be addressed.

### Current Status

**Experiment 0 feasibility: PASSED**

The basic phenomenon exists:

> Different update strategies applied to the same model state and new experience can produce meaningfully different acquisition–damage outcomes.

However, the stronger hypothesis of experience-dependent plasticity remains untested.

### Next Experiment

## Experiment 0.6 — Multi-Family Plasticity

Construct a new controlled-memory environment containing multiple task families, such as:

1. factual associations;
2. symbolic mappings;
3. rule-learning tasks;
4. arithmetic transformations.

Introduce new experiences from each family and repeat candidate-update comparisons.

Key questions:

1. Does the optimal update strategy change across task families?
2. Are different model regions safer or more plastic for different kinds of experiences?
3. Can gain–damage differences be explained by update magnitude alone?
4. Does lexical/token overlap explain the observed interference?
5. Is there enough experience-dependent variation to justify learning a plasticity fingerprint?

A positive result would provide the foundation for pre-update gain/damage forecasting.


## Experiment 0.6 — Multi-Family Plasticity

### Objective

Test whether different kinds of incoming experience produce different
gain–damage tradeoffs across candidate model-update strategies.

The experiment was motivated by Experiment 0.5, where late-layer
updating performed strongly on a single synthetic factual family.
The goal was to determine whether this behavior generalized across
different task families or whether experience type altered the optimal
plasticity strategy.

### Tokenization control

Four task families were constructed:

- factual associations
- symbolic mappings
- rule/classification associations
- synthetic arithmetic transformations

Each family contained:

- 5 protected memories
- 5 novel episodes

All 40 expected answers were audited using the GPT-2 tokenizer.

Result:

- Total answers: 40
- Single-token answers: 40
- Multi-token answers: 0

Novel and protected arithmetic target overlap was also removed before
training.

### Controlled multi-family memory

A new GPT-2 controlled-memory checkpoint was trained jointly on
20 protected memories (5 per family).

Training stopped only when every individual protected-memory loss was
below 0.05.

The target was reached at epoch 29.

Final protected losses:

- Facts: 0.0016
- Symbols: 0.0001
- Rules: 0.0013
- Arithmetic: 0.0165
- Overall average: 0.0049
- Worst individual memory: 0.0392

The checkpoint was saved as:

results/checkpoints/multifamily_controlled_memory

Protected-memory losses fluctuated before convergence, suggesting
non-trivial interaction between jointly stored memories.

### Candidate sweep

20 novel episodes × 6 candidate update strategies produced
120 candidate outcomes.

Candidates:

- full model, lr = 1e-6
- full model, lr = 1e-5
- full model, lr = 5e-5
- early layers 0–3, lr = 1e-5
- middle layers 4–7, lr = 1e-5
- late layers 8–11, lr = 1e-5

All candidates within an episode:

- started from the same controlled checkpoint
- used the same random seed
- received 10 update steps

Damage was measured over all 20 protected memories and separately for
each protected family.

### Aggregate results

Candidate               Mean Gain    Mean Damage
------------------------------------------------
full_lr_1e-6             1.903        0.00033
early_layers             4.813        0.00337
middle_layers            5.142        0.00898
late_layers              5.461        0.00272
full_lr_1e-5             8.609        0.08791
full_lr_5e-5             9.205        2.03190

Late-layer updating strictly dominated both early- and middle-layer
updating in aggregate, obtaining greater acquisition with lower mean
protected-memory damage.

The aggregate Pareto frontier consisted of:

- full_lr_1e-6
- late_layers
- full_lr_1e-5
- full_lr_5e-5

Late-layer updating therefore emerged as the strongest practical
low-damage default.

### Family-level behavior

Local differences remained:

- Facts:
  early gain = 5.719, damage = 0.00052
  late gain  = 5.625, damage = 0.00090

- Symbols:
  late layers provided the strongest conservative acquisition
  (gain = 6.035, damage = 0.00036).

- Rules:
  early layers produced slightly more acquisition
  (5.209 vs 5.169), but late layers caused substantially less damage
  (0.00082 vs 0.00213).

- Arithmetic:
  late layers provided the better gain–damage tradeoff.
  Middle layers produced only slightly more acquisition while causing
  more than three times as much damage.

### Candidate × family interaction

Two-way candidate × novel-family analysis did not produce a
statistically significant interaction.

Acquisition:
F(15,96) = 1.15, p = .324

Damage:
F(15,96) = 0.95, p = .512

Candidate choice explained substantially more variation than task
family:

- approximately 59% of acquisition variation
- approximately 93% of log-damage variation

Because only five novel episodes were available per family, these null
interactions are considered inconclusive rather than proof that family
effects do not exist.

### Cross-family interference

Protected-memory damage was strongly family structured.

Across all candidates, same-family damage was approximately 307× the
off-family mean.

After excluding destructive high-learning-rate full-model updates,
same-family damage remained approximately 28× larger than off-family
damage.

Arithmetic protected memories were the most vulnerable under
conservative updates, primarily due to arithmetic-to-arithmetic
interference.

### Static versus adaptive policies

Without a forgetting constraint, adaptation provided no advantage:
full_lr_5e-5 maximized acquisition for all episodes but caused severe
forgetting.

Under a conservative mean-damage budget of 0.01:

Best static policy:
late_layers
gain = 5.461
damage = 0.00272

Family-conditioned policy:
gain = 6.849
damage = 0.00409

This represents approximately 25% greater acquisition than the best
static conservative policy.

Hard per-episode constrained oracle:
gain = 7.087
damage = 0.00354

This establishes potential room for adaptive candidate selection under
a forgetting constraint, but does not yet demonstrate that such a
policy can be predicted from pre-update signals.

### Hypothesis verdict

The broad hypothesis that different experience families require
fundamentally different update locations was weakened.

The dominant empirical pattern is:

    late-layer updating provides a strong general low-damage default,
    while some experiences may justify deviations from that default.

The adaptive-plasticity hypothesis therefore survives in a narrower
form.

The relevant forecasting problem becomes:

    Given a candidate experience and several possible updates, can
    pre-update signals identify when deviating from the strong static
    default yields a superior constrained gain–damage tradeoff?

### Next step

Experiment 0.6b will replicate the constrained adaptive opportunity
with:

- more novel episodes per family
- denser early/late learning-rate candidates
- pre-registered damage budget D <= 0.01
- additional seed/checkpoint replication if the effect survives

Only after establishing a reproducible constrained oracle gap will
pre-update plasticity fingerprints be developed.

## Experiment 0.6b — Constrained Adaptive-Plasticity Replication

### Objective

Replicate whether adaptive candidate selection remains useful after
expanding the number of novel episodes and imposing a pre-registered
forgetting constraint.

The experiment focused on a narrower hypothesis than Experiment 0.6:

    A strong late-layer conservative default may exist, but some
    experiences may safely support more aggressive or alternative
    updates.

The primary decision criterion was a protected-memory mean-damage
budget of:

D <= 0.01

### Dataset and token controls

The protected checkpoint remained unchanged:

- 20 protected memories
- 5 each from facts, symbols, rules, and arithmetic

The novel set was expanded to:

- 10 factual episodes
- 10 symbolic episodes
- 10 rule/classification episodes
- 10 arithmetic episodes
- 40 novel episodes total

A stricter tokenizer audit was performed before the sweep.

Results:

- Total protected + novel answers: 60
- Single-token answers: 60
- Multi-token answers: 0
- Protected/novel target-token collisions: 0
- Within-family novel duplicates: 0
- Cross-family novel target duplicates: 0

The token audit therefore passed cleanly.

### Candidate grid

Eight candidate update strategies were evaluated:

- full_lr_1e-6
- full_lr_1e-5
- early_lr_5e-6
- early_lr_1e-5
- early_lr_2e-5
- late_lr_5e-6
- late_lr_1e-5
- late_lr_2e-5

All candidates for a given episode:

- started from the same controlled checkpoint
- used the same episode-specific random seed
- received the same number of update steps

Total outcomes:

40 episodes × 8 candidates = 320

Results were saved to:

results/experiment_0/constrained_replication_candidates_seedcontrolled.csv

### Main constrained-policy result

Under the pre-registered mean-damage budget D <= 0.01:

Best aggregate conservative static candidate:

late_lr_1e-5

Mean acquisition gain:
5.331

Mean protected-memory damage:
0.00200

Per-episode constrained oracle:

Mean acquisition gain:
8.261

This is approximately 54.9% higher acquisition than the best
aggregate static conservative policy.

The pre-registered criterion for strong adaptive opportunity was an
oracle improvement of approximately 10% or more.

The observed adaptive opportunity therefore exceeded the threshold by
a wide margin.

### Strict hard-cap interpretation

Although late_lr_1e-5 was safe on average, it exceeded the D <= 0.01
damage cap on 2 of 40 episodes.

If a static policy is required to satisfy the hard cap on every
episode, the best fully safe static candidate becomes:

late_lr_5e-6

Mean acquisition gain:
3.080

Maximum episode damage:
0.00433

Relative to this strict static baseline, the constrained per-episode
oracle achieves approximately 168% greater acquisition.

Both average-budget and hard-cap comparisons should be reported
separately in later analysis.

### Oracle candidate diversity

The constrained oracle did not collapse to a single update strategy.

Oracle selections across 40 episodes:

- full_lr_1e-5: 20
- late_lr_2e-5: 9
- late_lr_1e-5: 7
- early_lr_2e-5: 2
- early_lr_1e-5: 1
- late_lr_5e-6: 1

This demonstrates genuine episode-level variation in the best safe
candidate.

### Family-level oracle behavior

The constrained oracle showed different selection patterns across
families.

Facts:
full_lr_1e-5 selected in 8/10 episodes.

Symbols:
full_lr_1e-5 selected in 8/10 episodes.

Rules:
late_lr_2e-5 selected in 6/10 episodes.

Arithmetic:
late_lr_1e-5 selected in 7/10 episodes.

The dominant adaptive signal therefore appears to involve safe update
intensity more strongly than a simple family-to-layer routing rule.

### Factual early-layer exception

The factual early-layer effect observed in Experiment 0.6 replicated
only weakly.

At some lower learning rates, early layers produced slightly greater
mean acquisition and slightly lower mean damage than late layers.

However, the effect was not consistent across individual factual
episodes and disappeared at the more aggressive learning rate.

Conclusion:

The factual early-layer effect should not be treated as a robust
routing rule.

### Arithmetic vulnerability

Arithmetic protected memories again emerged as the most vulnerable
destination under conservative updating.

Arithmetic-to-arithmetic interference increased sharply with update
intensity.

Representative same-family arithmetic damage:

- full_lr_1e-6: 0.00185
- late_lr_5e-6: 0.00386
- early_lr_5e-6: 0.00450
- late_lr_1e-5: 0.02332
- early_lr_1e-5: 0.02339
- early_lr_2e-5: 0.21958
- late_lr_2e-5: 0.32909
- full_lr_1e-5: 0.90205

Among conservative candidates, arithmetic memories were substantially
more vulnerable than facts, symbols, or rules.

Same-family positive damage remained approximately 13× larger than
off-family damage.

### Interaction interpretation

The larger replication did not show a strong candidate × family
interaction for acquisition.

However, candidate × family structure became much stronger for damage.

This suggests:

    Experience family may not strongly determine how much can be
    learned, but it can strongly affect how damaging a candidate
    update will be.

This supports forecasting protected-memory risk rather than simple
family-based update routing.

### Hypothesis verdict

The broad hypothesis:

    different experience families require fundamentally different
    parameter locations

is not supported strongly enough to pursue.

The narrower constrained-adaptation hypothesis is strongly supported:

    A strong late-layer default exists, but the maximum safe update
    intensity varies substantially across experiences.

The central forecasting problem is therefore:

    Given a candidate experience and candidate update, can pre-update
    signals predict acquisition gain and protected-memory damage well
    enough to select the highest-gain update satisfying D <= 0.01?

### Decision

Experiment 0.6b passes the adaptive-opportunity gate.

The project will now proceed to:

Experiment 0.7 — Pre-update Plasticity Fingerprints

The goal is to measure signals available before updating the model and
test whether they can forecast candidate-specific gain and damage.

## Final Robustness Phase — Ablation, Seed Robustness, and Cross-Model Replication

### Fingerprint ablation

The forecasting fingerprint was decomposed into:

- candidate metadata only
- gradient magnitude/profile + candidate
- gradient alignment + candidate
- full fingerprint + candidate

Evaluation used grouped cross-validation by episode.

For acquisition forecasting:

- Candidate only:
  R² ≈ 0.464
- Alignment + candidate:
  R² ≈ 0.545
- Magnitude/profile + candidate:
  R² ≈ 0.756
- Full fingerprint + candidate:
  R² ≈ 0.764

Magnitude/profile features therefore contain most of the predictive
signal for acquisition.

For damage forecasting:

- Candidate only:
  R² ≈ 0.117
- Magnitude/profile + candidate:
  R² ≈ 0.136
- Alignment + candidate:
  R² ≈ 0.455
- Full fingerprint + candidate:
  R² ≈ 0.430

Gradient alignment carries substantially more damage-predictive
information than raw gradient magnitude/profile.

This supports a differentiated interpretation:

    Gradient magnitude/profile is primarily informative about
    acquisition potential.

    Gradient alignment with protected memories is primarily informative
    about interference and protected-memory damage.

The full fingerprint is not always optimal for damage forecasting,
suggesting that target-specific feature subsets may be preferable.

### Seed and split robustness

The grouped forecasting experiments were repeated across multiple
episode splits and Random Forest seeds.

Across 10 repeated grouped evaluations:

Acquisition R²:

- candidate only: ~0.422
- magnitude/profile + candidate: ~0.766
- alignment + candidate: ~0.501
- full fingerprint + candidate: ~0.776

Damage R²:

- candidate only: ~0.111
- magnitude/profile + candidate: ~0.141
- alignment + candidate: ~0.405
- full fingerprint + candidate: ~0.355

The ranking of feature groups remained stable across repeated splits.

Therefore, the main fingerprint conclusions are not attributable to a
single favorable train/test partition or Random Forest seed.

### Second-model replication — DistilGPT-2

A compact second-model replication was performed using DistilGPT-2.

#### Controlled-memory setup

DistilGPT-2 was trained on 12 protected memories.

The controlled-memory target required every protected answer loss to
fall below 0.05.

The target was reached after 50 epochs.

Final statistics:

- overall mean protected loss: ~0.00592
- worst protected item loss: ~0.04133
- arithmetic remained the hardest protected family

#### Candidate-update replication

12 novel episodes × 6 candidate strategies produced 72 outcomes.

Aggregate examples:

- full_lr_1e-5:
  gain ≈ 10.149
  damage ≈ 0.13337

- early_lr_1e-5:
  gain ≈ 6.592
  damage ≈ 0.00922

- late_lr_1e-5:
  gain ≈ 5.638
  damage ≈ 0.00125

- late_lr_5e-6:
  gain ≈ 3.694
  damage ≈ 0.00049

The aggregate Pareto frontier contained:

- late_lr_5e-6
- late_lr_1e-5
- early_lr_1e-5
- full_lr_1e-5

This confirms that candidate-dependent acquisition–damage tradeoffs
are not unique to GPT-2 small.

Arithmetic protected memories again exhibited the greatest
vulnerability under conservative updates.

### Expanded DistilGPT-2 forecasting replication

The initial 12-episode forecasting experiment was judged underpowered.

The novel set was therefore expanded to:

- 24 novel episodes
- 6 candidates per episode
- 144 candidate outcomes

Evaluation used repeated GroupShuffleSplit by episode:

- 20 splits
- 25% held-out episodes per split

Acquisition forecasting:

Candidate only:
- MAE ≈ 2.019
- R² ≈ 0.421

Magnitude/profile + candidate:
- MAE ≈ 1.715
- R² ≈ 0.477

Alignment + candidate:
- MAE ≈ 2.227
- R² ≈ 0.304

Full fingerprint + candidate:
- MAE ≈ 1.766
- R² ≈ 0.476

Magnitude/profile therefore again improved acquisition forecasting
relative to candidate identity alone.

Damage forecasting remained substantially weaker.

Candidate only:
- MAE ≈ 0.0302
- mean R² < 0

Magnitude/profile + candidate:
- MAE ≈ 0.0234
- mean R² < 0

Alignment + candidate:
- MAE ≈ 0.0205
- R² ≈ 0.132

Full fingerprint + candidate:
- MAE ≈ 0.0224
- mean R² < 0

Alignment was again the most useful fingerprint component for damage,
but cross-model damage forecasting remained noisy and substantially
weaker than on GPT-2 small.

### Cross-model verdict

Qualitative cross-model replication passed.

The following findings replicated on DistilGPT-2:

1. Candidate update choice strongly changes the acquisition–damage
   tradeoff.

2. Conservative/localized updates provide substantially lower
   protected-memory damage than aggressive full-model updates.

3. Gradient magnitude/profile features improve acquisition forecasting
   beyond candidate identity alone.

4. Gradient alignment provides the strongest damage-predictive signal
   among the tested fingerprint components.

However, robust cross-model damage forecasting was not established.

The paper should therefore distinguish between:

    robust qualitative transfer of the plasticity phenomenon

and

    weaker cross-model transfer of quantitative damage prediction.

### Final experimental status

The planned pre-paper experimental program is complete.

The evidence supports proceeding to paper writing after freezing the
final claims and limitations.

## External Validation — Qwen2.5-0.5B on RippleEdits

**Model:** Qwen2.5-0.5B  
**Benchmark:** RippleEdits real-world-entity edits  
**Edits:** 100 total (34 random, 33 recent, 33 popular)  
**Evaluation:** 5-fold `GroupKFold` grouped by `edit_id`

### Objective and evaluation protocol

This experiment tests whether pre-update plasticity fingerprints transfer
to a modern Qwen architecture and a benchmark based on real-world entities.
The protected criterion was evaluated only on baseline-known items, defined
as NLL/token <= 4.0. Ripple probes were excluded from the damage measure.

An initial one-step sweep and a 20-edit strength-calibration run were used
only to calibrate the final candidate grid; they are not part of the final
forecasting evaluation. The final sweep contains 400 candidate outcomes:
100 edits x 4 candidates.

### Final candidate grid and aggregate outcomes

| Candidate | Acquisition gain (G) | Known-protected damage (D_known) | Stable edits |
|---|---:|---:|---:|
| `early_lr_1e-5_steps10` (layers 0–7) | 4.4428 | -0.1459 | 100/100 |
| `full_lr_1e-5_steps1` | 2.4010 | -0.1954 | 100/100 |
| `full_lr_2e-5_steps5` | 5.9546 | +0.4801 | 92/100 |
| `late_lr_1e-5_steps10` (layers 16–23) | 5.3135 | -0.2260 | 100/100 |

The aggressive full-model candidate attained the largest mean acquisition
gain but was non-finite on 8 of 100 edits and had positive mean
known-protected damage. The localized early and late candidates were stable
on all edits and had negative mean known-protected damage.

### Grouped forecasting results

All forecasts were evaluated out of fold using 5-fold `GroupKFold` by
`edit_id`.

| Feature set | Acquisition R² | Acquisition MAE | Damage R² | Damage MAE |
|---|---:|---:|---:|---:|
| Candidate only | 0.2226 | 1.8043 | 0.0151 | 0.6075 |
| Magnitude/profile + candidate | 0.8715 | 0.6627 | -0.2834 | — |
| Alignment + candidate | 0.0262 | 2.0355 | -0.3255 | — |
| Full fingerprint + candidate | 0.8710 | 0.6663 | -0.2321 | — |

Gradient magnitude/profile features transferred strongly for forecasting
acquisition. In contrast, no tested feature set produced useful
out-of-fold damage forecasting on this benchmark. In particular, this
experiment does not support a claim that Qwen damage forecasting succeeded.

### Acquisition-only selection analysis

The selection summary was computed after correcting the stability-denominator
metric: realized and oracle gains are compared on the same stable selected
edits. Acquisition-only selection largely collapses to the aggressive full
candidate. Candidate-only selection chooses `full_lr_2e-5_steps5` for all
100 edits; magnitude/profile and full-fingerprint selection each choose it
for 97/100 edits and choose the late candidate for the remaining 3. The 8%
instability of the aggressive candidate therefore remains.

These are acquisition-only policies, not safety-constrained selection
results. They should not be used to claim safe update routing or successful
damage-aware selection.

### Interpretation

This external validation provides evidence that pre-update gradient
magnitude/profile features can transfer for acquisition forecasting to a
modern Qwen architecture on a real-world-entity benchmark. It does not show
that damage forecasts transfer. Collateral-damage generalization is therefore
the key remaining bottleneck for safety-aware test-time plasticity.
