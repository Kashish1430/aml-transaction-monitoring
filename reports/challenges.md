# Engineering Challenges & Decisions Log

A running record of the non-obvious problems hit while building this project, what
actually caused each one, and what fixed it — kept as source material for talking about
this project in interviews. Unlike `results.md` (headline metrics) this is a narrative
log, not a regenerable artifact; it's updated by hand as real problems get solved.

Format per entry: **what broke**, **root cause** (not just the symptom), **the fix**,
**why it's a good interview story**. A few entries are design/architecture questions
rather than bugs — those use **the question** / **the answer** / **why it matters**
instead, since nothing "broke."

---

## Architecture — why this is a batch pipeline, not a real-time scoring service

- **The question:** After Phase 3's batch feature pipeline (`data/processed/features.parquet`,
  built once from the full historical dataset), the natural follow-up is: what happens
  when a *new* transaction arrives? Surely a real bank doesn't rebuild the whole account
  graph and rescan years of history for every incoming wire — that would be an absurd
  amount of computation per transaction. So how does this actually work in production,
  and is "batch" just a simplification for a portfolio project, or is it how real AML
  transaction monitoring is actually architected?
- **The answer:** Real banks run two structurally different kinds of controls, and it's
  easy to conflate them:
  1. **Real-time, blocking — sanctions/watchlist screening.** Before a wire completes,
     the counterparty is checked against OFAC/sanctions lists. This has to be real-time
     and has to be able to block the transaction, because letting funds reach a
     sanctioned entity even briefly is its own severe violation. It's a lookup problem
     (is this name/account on a list?) with no historical context required, which is
     exactly why real-time is both necessary and cheap here.
  2. **Batch/periodic, non-blocking — AML typology detection.** This is what this
     project models (structuring, layering, fan-in/fan-out, cycles). These typologies
     are only visible as *patterns across multiple transactions over time* — a single
     transaction essentially never looks suspicious in isolation. Detecting them
     requires a rolling window of history, which is why real transaction monitoring
     systems commonly re-score on a nightly/end-of-day batch cadence rather than a
     stateless per-transaction API call — the pattern-based nature of the problem
     forces the batch shape, not just convenience.

  This lines up with the regulatory timeline, which is itself batch-tolerant, not
  real-time: an alert leads to analyst investigation, and if there are grounds to
  suspect laundering, the institution files a **SAR (Suspicious Activity Report)** with
  the regulator (FinCEN in the US) — typically required within 30 days of detection,
  extendable to 60. Separately, any transaction over $10,000 gets a **CTR (Currency
  Transaction Report)** filed regardless of suspicion, a routine threshold-based filing
  (this is what `rules_baseline.py`'s large-amount rule models) — a SAR and a CTR are
  two different regulatory instruments, not the same thing under different names.

  None of this means a new transaction is literally unscored until the next nightly
  batch, though — production systems handle it with *incremental* state, not by
  re-running this project's batch pipeline per transaction:
  - Rolling counts/sums: a small per-account rolling log (feature store / cache),
    updated in O(1) per new transaction — not the `merge_asof`-based batch approach
    built here for scoring millions of historical rows at once.
  - Graph degree/fan-in/fan-out: the account graph already exists in memory or a graph
    DB; a new edge just increments a couple of counters, with the same FIFO-pruning
    idea used in `build_daily_graph_features` (drop edges older than the lookback)
    running continuously instead of once per historical day.
  - Cycle detection: never re-run `nx.simple_cycles` over the whole graph for one new
    edge. Ask one bounded question instead — "starting from B, is there a path back to
    A within `cycle_max_length - 1` hops?" — a local traversal from a single node,
    genuinely millisecond-scale, not a full-graph enumeration.
- **Why it matters:** this project's "alert triage, not live blocking" framing (see
  `CLAUDE.md`'s project description) isn't a simplification to apologize for — it's the
  architecturally correct call for *this specific problem*, because typology detection
  is inherently a windowed/relational pattern-matching problem operating on a 30-60 day
  regulatory timeline, not a millisecond decision problem the way sanctions screening
  is. Being able to explain *why* batch is correct here — not just that it's what got
  built — is what separates "I trained a model on a Kaggle dataset" from "I understand
  how this fits into a real compliance operation."

---

## Phase 1 — Data acquisition & validation

### Prevalence was 20x more extreme than the brief assumed
- **What broke:** Nothing broke, but a planning assumption did. The original project
  brief estimated ~2% laundering prevalence; the actual HI-Small data is ~0.10%
  (5,177 positives out of 5,078,345 transactions).
- **Root cause:** The brief's number was a generic estimate, not measured against the
  actual downloaded files.
- **The fix:** Measured it directly in `notebooks/01_eda.ipynb` via
  `data_loader.dataset_stats` before any modelling decision was made, and propagated the
  real number into every later document (`CLAUDE.md`, `PLAN.md`) instead of quietly
  working around the discrepancy.
- **Interview angle:** Verify assumptions against real data before designing around
  them — at 20x more extreme imbalance than planned, accuracy is even more
  meaningless as a metric than the brief anticipated, which directly justified the
  precision@k / recall-per-typology / FP-reduction-vs-baseline evaluation strategy used
  throughout the rest of the project.

### No transaction ID anywhere in the dataset
- **What broke:** `HI-Small_Patterns.txt` labels which transactions belong to which of
  the 8 laundering typologies, but there's no shared key to join it back onto
  `HI-Small_Trans.csv` — the file isn't even a CSV, it's blocks of
  `BEGIN LAUNDERING ATTEMPT - <TYPOLOGY>` / raw rows / `END LAUNDERING ATTEMPT`.
- **Root cause:** Dataset design choice by the publisher, not something fixable.
- **The fix:** `data_loader.load_patterns` parses the block structure and joins back
  onto transactions by matching the full set of shared field values (timestamp, banks,
  accounts, amounts, format). This only recovers 3,209 of 5,177 laundering transactions
  (62%) — the rest are laundering but don't match a named typology block.
- **Interview angle:** Know the difference between "my parser is broken" and "the data
  genuinely doesn't support what I'm asking of it." The 38% gap is disclosed everywhere
  (`CLAUDE.md`, this log) as a data limitation, not silently patched over or hidden.

### Duplicate `Account` column header
- **What broke:** `HI-Small_Trans.csv` has two columns both literally named `Account`
  (sender and receiver); pandas silently auto-renames the second one to `Account.1` on
  load, which is easy to miss if you don't inspect the raw columns first.
- **The fix:** Explicit rename mapping in `data_loader.TRANSACTION_RENAME` documents
  which physical column is sender vs. receiver, so the ambiguity is resolved once, in
  one place, rather than every downstream script having to know about it.

---

## Phase 2 — Rules baseline

### A flat USD threshold would have been silently wrong
- **What broke:** Nothing yet — caught before it became a bug. The large-amount and
  structuring rules need a single USD-equivalent threshold, but the dataset has 15
  payment currencies (including Yen, Rupee, and Bitcoin) at wildly different numeric
  scales.
- **Root cause:** A naive threshold on raw `amount_paid` would treat 10,000 Yen
  (~$71) the same as 10,000 US Dollars — checked empirically (per-currency amount
  distributions) before writing the rule, not discovered after the fact.
- **The fix:** `rules_baseline.add_usd_amount` converts every amount to USD via a
  static FX table (`config.yaml`'s `fx_rates_to_usd`) before any thresholding.
  Documented as a deliberate simplification (a real deployment would use point-in-time
  FX rates per transaction date, not one static table for the whole dataset).

---

## Phase 3 — Feature engineering

This phase had the two most substantial engineering problems in the project so far —
both performance, both caught by profiling against the real data rather than trusting
small hand-built test cases to represent full-scale behaviour.

### 1. Distinct-counterparty rolling window was 30-40x slower than it needed to be
- **What broke:** The fan-in/fan-out distinct-counterparty feature
  (`_rolling_distinct_counterparties`) needs a genuine sliding-window two-pointer scan
  per account (distinct-count isn't a simple prefix-sum difference like count/sum are,
  since a counterparty can leave and re-enter the window). The first implementation
  grouped both the event history and the query rows with `DataFrame.groupby("account_key")`
  per account.
- **Root cause:** In an 18-day window, most accounts only have 1-2 transactions total —
  so almost every "group" `groupby` constructed was a single-row DataFrame. The
  per-group pandas object-construction overhead (not the two-pointer logic itself,
  which is O(1) amortized per row) dominated the runtime. Confirmed by isolating the
  two helper functions and timing them separately on real-data subsets:
  `_rolling_count_sum` (vectorized, no groupby) ran in 0.03-0.11s on 5k-20k rows, while
  `_rolling_distinct_counterparties` took 5.15s-19.91s on the same subsets — roughly
  1ms of pure overhead per row, almost none of it useful work.
- **The fix:** Rewrote it to sort both frames once by `(account_key, timestamp)`, find
  account-group boundaries with `np.unique`/`np.searchsorted`, and slice raw numpy
  arrays directly inside the two-pointer loop — no `DataFrame.groupby` anywhere in the
  hot path. Same inputs, same outputs (all 31 tests still pass), but 5.15s -> 0.16s
  (32x) at 5k rows and 19.91s -> 0.49s (40x) at 20k rows.
- **Interview angle:** "Profile before you optimize" cliché made concrete — the
  bottleneck wasn't the algorithm's asymptotic complexity (it really was O(n) per
  account), it was a constant-factor library overhead invisible until you time the
  actual library calls, not just eyeball the big-O.

### 2. My own performance benchmarking methodology was misleading
- **What broke:** Before running the full pipeline, I benchmarked
  `build_daily_graph_features` on row-count prefixes of the data (first 50k, 200k
  rows) and it looked fast (0.86s-3.71s) — false confidence. The actual full-dataset
  run in the notebook then hung and hit a 40-minute cell timeout.
- **Root cause:** HI-Small's transaction volume is extremely front-loaded in time: day
  1 alone has 1,114,921 of the dataset's 5,078,345 transactions (~22%), and days 1-2
  combined are ~1.9M (~37%). Any row-count prefix under ~1.9M rows never leaves day 1,
  and graph features for day 1 are trivially cheap (empty graph, cold start, nothing
  accumulated yet) *by construction* — the subsample was accidentally testing only the
  cheapest possible case, regardless of how many rows it contained.
- **The fix:** Re-profiled using calendar-day counts instead of row counts (first 3,
  6, 9 days), which exposed the real growth curve immediately.
- **Interview angle:** A benchmark's sampling strategy is itself a hypothesis about the
  data that needs checking — "more rows" and "more realistic" are not the same thing
  when the underlying distribution is skewed. This is the kind of bug that only shows
  up in production-scale runs, which is exactly why the notebook was executed
  end-to-end against the full dataset as this phase's actual completion check, not left
  to "should be fine" extrapolation from a subsample.

### 3. An unbounded "all history to date" account graph never stopped growing
- **What broke:** The original design (graph built from every transaction since day 1,
  re-used and extended day by day) showed a clearly worsening cost curve once profiled
  correctly by day: 73.25s for the first 3 days, 168.92s for 6 days, 521.11s for just
  9 of the dataset's 18 days — and climbing, not plateauing.
- **Root cause:** Two compounding factors. First, the data's front-loaded volume (see
  above) meant the graph was already large after only 1-2 days and kept being extended
  every subsequent day without ever shrinking. Second, `nx.simple_cycles` cost scales
  with the graph's edge count, so a same day's cycle-detection call got more expensive
  every day the unbounded graph grew, for the entire 18-day span.
- **First fix attempted (and rejected): rebuild the graph from scratch each day from a
  bounded 7-day trailing window.** This correctly capped the maximum graph size, but
  was measured to be *slower* than the unbounded version for the early days (241.49s vs.
  168.92s for the first 6 days) — because for any two days within 7 days of each
  other, the same edges get re-inserted into a brand-new graph object on both days,
  instead of being inserted once and reused. Bounding size isn't free if the bounding
  mechanism itself does redundant work.
- **The actual fix: incremental pruning.** Kept a single running graph across the
  whole loop, added each edge exactly once, and tracked which day each edge was added
  on in a FIFO queue; before processing each day, edges more than `lookback_days` old
  are explicitly removed (`MultiDiGraph.remove_edge` by stored edge key) rather than
  the graph being thrown away and rebuilt. This amortizes to O(1) insert + O(1) removal
  per edge over the whole run, while still keeping the graph bounded to at most
  `lookback_days` worth of history at any point. Also reduced `cycle_max_length` from
  4 to 3 (shorter cycles are both the more common laundering signature and cheaper to
  enumerate) as a second, independent lever.
- **Outcome:** Verified on the full 5,078,345-row, 18-day dataset:
  `build_daily_graph_features` completed in 463.64s (~7.7 min), plateauing rather than
  continuing to worsen. Combined with `assemble_feature_table`'s 901.10s (~15 min), the
  full `build_modelling_table` pipeline finished in ~1532s (~25.5 min) — down from a
  design that, extrapolated from its own growth curve, would have taken many hours (if
  it terminated at all) on the full dataset.
- **Interview angle:** This is a three-part story, which is what makes it a strong one:
  (1) a benchmarking blind spot hid the real problem, (2) the "obviously correct" first
  fix (bound the window) was empirically *worse* than the naive version for a
  non-obvious reason (redundant work, not wrong complexity class), and (3) the actual
  fix required recognizing that "bounded size" and "no redundant work" are two separate
  requirements that need two separate mechanisms (a size cap alone isn't enough; it's
  the incremental add-once/remove-once discipline that made it fast). Also a clean
  example of turning a hard performance constraint into a documented, defensible
  modelling choice rather than an apologetic workaround: a 7-day lookback is *also* the
  more honest modelling assumption, since a fan-in/fan-out/cycle motif from 15 days ago
  is stale signal for typologies that typically play out over hours to days.

---

## Phase 4 — Modelling

### The train/val/test split surfaces a real distribution shift instead of hiding it
- **What happened:** A strictly time-ordered 60/20/20 split (`time_ordered_split`)
  produces train/val/test prevalence of 0.075% / 0.107% / 0.177% respectively — the
  test split has more than double train's laundering rate.
- **Root cause:** Not a bug — HI-Small's last several days have very low transaction
  volume but a disproportionately high concentration of laundering activity in what
  remains (already flagged in Phase 1/3 investigation — see `CLAUDE.md`'s Dataset
  section). A time-ordered split, by construction, doesn't average this away the way a
  random split would.
- **The fix:** Nothing to fix — this is the correct, honest behavior. `scale_pos_weight`
  is deliberately computed from the *training* split only (never val/test), so this
  drift can't leak into how the model is weighted; it's disclosed directly in
  `reports/results.md`'s split table instead of being smoothed over.
- **Interview angle:** A time-aware split is only doing its job if it's allowed to
  surface uncomfortable facts about how the data behaves over time — a model that looks
  great on a random split but was never actually tested against a shifted future
  distribution is the more dangerous outcome. This is also a concrete illustration of
  why the project brief insists on time-ordered splits over random k-fold: a random
  split here would have masked a distribution shift a production deployment would
  actually have to face.

### `payment_format_ACH` dominates feature importance — investigated, not just reported
- **What happened:** In the first trained model, `payment_format_ACH`'s gain-based
  importance (145,749.8) is roughly 3.5x the next-highest feature — a big enough gap to
  be suspicious rather than simply reported as "the top feature."
- **Root cause, found by checking rather than assuming:** 86.6% of all laundering
  transactions in the dataset use ACH format; ACH's laundering rate (0.75%) is ~7.3x the
  dataset's overall prevalence, while `Wire` and `Reinvestment` have *zero* laundering
  transactions anywhere in the data. This is either a genuine signal (ACH is a real,
  commonly-abused layering channel in practice) or an artifact of how IBM's AMLworld
  generator constructs its laundering scenarios — the data alone can't fully
  distinguish the two, which is itself the point worth disclosing rather than picking
  the more flattering interpretation.
- **The fix (an ablation, not a removal):** Retrained the identical model with every
  `payment_format_*` column dropped, to quantify — not just assert — how much of the
  model's lift depends on it. Result: test PR-AUC falls from 0.3967 to 0.0705, ROC-AUC
  from 0.9828 to 0.9160, precision@100 from 92% to 62%. `payment_format` really is
  carrying a large share of the headline numbers. But the ablated model still performs
  far better than random (62% precision@100 vs. a 0.18% base rate), and — the more
  important result for this project's actual thesis — the features that rise to the
  top once `payment_format` is removed are `receiver_in_30d_distinct_counterparties`,
  `sender_graph_in_cycle`, and `sender_out_7d_distinct_counterparties`: exactly the
  Phase 3 account/window and graph features, not noise.
- **Interview angle:** Noticing a suspiciously dominant feature and *investigating* it
  (checking the label correlation directly, then quantifying the dependency via an
  ablation) rather than either ignoring it or silently dropping it, is the difference
  between reporting a number and understanding it. It also produces a more honest,
  more interesting headline than the unqualified PR-AUC would have: "the categorical
  payment-channel signal does a lot of work in this synthetic dataset and that's
  disclosed plainly, but the network/behavioral features this project is actually about
  hold up on their own once it's removed" is a stronger, more defensible claim than a
  single unexamined AUC number.

### Isolation Forest and XGBoost agree on literally nothing — and that's diagnosable
- **What happened:** Isolation Forest, trained unsupervised on the same feature set,
  has **zero** overlap with XGBoost's top-1000 test-set alerts, and catches only 1 of
  1,797 actual laundering transactions in its own top-1000 — far worse than a result
  that would even suggest "different but complementary."
- **Root cause, found by inspecting the actual flagged transactions rather than just
  the overlap number:** Isolation Forest's top-1000 sit at the extreme tail of raw
  volume features — mean `sender_out_30d_count` of 149,867 against an overall test-set
  mean of 7,890 (near the dataset's actual maximum of 168,672), and similarly extreme
  `sender_out_30d_distinct_counterparties` and `amount_paid_usd`. It isn't finding
  laundering-specific behavior at all — it's rediscovering the handful of
  highest-throughput hub accounts (almost certainly legitimate high-volume businesses
  or bank-internal accounts), because isolation-based outlier detection has no concept
  of "extreme but legitimate" vs. "extreme and suspicious": it only measures how easy a
  point is to isolate via random partitioning, and heavily right-skewed raw count/
  amount features make the top of that skew trivially easy to isolate regardless of
  label.
- **The fix:** Not implemented in this phase, deliberately — noted as future work
  instead, to keep Phase 4's scope disciplined per the project brief's "resist boiling
  the ocean" guardrail: log-transforming heavy-tailed features before fitting Isolation
  Forest, or fitting it on ratio/score features (which don't have this raw-scale
  problem) instead of raw counts, would be the next thing to try.
- **Interview angle:** A disappointing headline number ("0% agreement") is much less
  interesting than the diagnosis behind it. This is a genuinely common, well-understood
  failure mode when applying generic multivariate outlier detection directly to
  heavy-tailed tabular features without addressing scale first — being able to name
  *why* an unsupervised layer failed in a specific, mechanistic way (not just "it didn't
  work") is a stronger signal of understanding than a clean agreement number would have
  been on its own. It's also a genuine, on-brief finding: the project brief frames the
  unsupervised layer's value as showing "where it agrees and disagrees" with the
  supervised model, and a 0%-overlap, hub-account-chasing result *is* that finding, not
  a failure to produce one.

### Memory headroom shaped how the feature matrix gets built
- **What happened:** `data/processed/features.parquet` (5.08M rows, 52 columns) occupies
  ~2.85GB in memory once loaded as float64, and the development machine's available
  memory was tight enough (checked via `psutil` before committing to an approach) that
  this mattered for how `prepare_feature_matrix` and `time_ordered_split` were written,
  not just as an afterthought.
- **The fix:** Two deliberate choices, both load-bearing rather than cosmetic: engineered
  features are downcast to float32 in `prepare_feature_matrix` (halves memory with no
  meaningful precision loss for count/ratio/degree-style features), and
  `time_ordered_split` skips `DataFrame.sort_values` entirely when the timestamp column
  is already monotonic (true for this pipeline's output, verified via
  `is_monotonic_increasing` rather than assumed) — avoiding an unnecessary full-table
  copy on top of the 2.85GB already resident.
- **Interview angle:** Not every performance decision needs a dramatic before/after
  benchmark to be worth making deliberately — checking actual system headroom before
  writing memory-hungry code, and avoiding an operation that's a no-op on this
  pipeline's real data rather than writing "obviously correct" generic code and hoping,
  is the same discipline as the Phase 3 performance work, just applied preemptively
  instead of reactively.

---

## Phase 5 — Evaluation

### The typology-label join was a forward-reference nothing had actually built
- **What happened:** `load_patterns`'s docstring, written back in Phase 1, said "see
  PLAN.md Phase 3 for where that join is implemented" — the join being how a
  transaction gets tagged with its typology (fan-in, cycle, ...), since the dataset has
  no transaction ID to key on. When Phase 5's recall-per-typology needed exactly this
  join, it turned out Phase 3 never actually built it — `build_modelling_table` only
  ever carried `is_laundering` through, not `pattern_type`. A stale forward-reference
  had been sitting uncorrected for two phases.
- **Root cause:** Phase 3's scope was account/window and graph features, which don't
  need typology labels — nothing in that phase's own work ever required the join, so a
  docstring's aspirational pointer never got checked against what actually got built.
- **The fix:** Implemented `data_loader.join_pattern_types`, matching `load_transactions`
  and `load_patterns` output on every field the two tables share (timestamp, banks,
  accounts, amounts, currencies, format, label). Verified empirically before trusting
  it: zero duplicate join keys on either side of the real data, and it recovers
  `pattern_type` for exactly 3,209 of 5,177 laundering transactions — matching
  `CLAUDE.md`'s previously-documented 62% figure precisely, which is itself a good
  cross-check that the join is doing the right thing. Also corrected the stale
  docstring to point here instead of leaving it wrong for whoever reads it next.
- **Interview angle:** Docstrings and comments that reference "where X is handled" are
  claims, not guarantees — they can rot the same way code can, just silently, because
  nothing fails until someone actually needs the thing being pointed at. Checking
  `grep`-for-real rather than trusting the comment is what caught this before it became
  a confused debugging session in Phase 5 instead of a five-minute correction.

### Comparing the model against the rules baseline required re-deriving the baseline, not reusing Phase 2's number
- **What happened:** Phase 2's `reports/results.md` reports the rules baseline's
  performance on the *full* dataset (36.3% alert rate, 60.6% recall). Phase 5 needs to
  compare the model against that baseline "at equal recall" — but the model is only
  ever evaluated on its *test split* (the last ~20% of transactions by time), a
  different, smaller, differently-composed population than the full dataset the Phase
  2 number describes.
- **Root cause (caught before implementing, not after):** using Phase 2's full-dataset
  60.6% recall figure directly against the model's test-split alert count would compare
  two different populations — not a fair "equal recall" comparison at all. And
  recomputing the rules on an isolated test-only slice would introduce a different bug:
  `rules_baseline.py`'s structuring and pass-through rules use rolling time windows
  (24h, 2h) that look at an account's *prior* transactions, which for rows near the
  start of the test split live in the training period — exactly the cold-start problem
  Phase 3's leakage-safety work exists to avoid, just reappearing at a different split
  boundary.
- **The fix:** Ran `apply_rules_baseline` on the **full** dataset first — so every rule
  keeps its complete rolling-window history right up to the test split's start — then
  restricted the result to the test split's rows by index (`baseline_full.loc[test_df.index]`,
  safe because `time_ordered_split` preserves original row positions as index values
  rather than resetting per split). This gives a rules-baseline number computed on
  *exactly* the model's test population, with no cold-start artifact. Result: the
  rules baseline's test-split recall is 68.8%, not Phase 2's 60.6% — a real,
  expected difference (the test split's own composition skews differently, per Phase
  4's split table), disclosed directly in `results.md` rather than silently
  reconciled or ignored.
- **Interview angle:** "Fair comparison" isn't just "same metric" — it's "same
  population, computed the same way, with the same information available at the same
  point in time." Two different, individually-defensible mistakes were available here
  (reuse a number from a different population, or recompute correctly-scoped but with
  a fresh cold-start bug) and neither is obviously wrong until you think through what
  each rule actually needs to see. Getting this right on the first attempt, by tracing
  through what each rule depends on before writing the comparison code, is the same
  discipline as Phase 3's leakage tests — just applied to a cross-phase comparison
  instead of a single feature.

### The model's scores aren't calibrated probabilities — diagnosed, not just plotted
- **What happened:** The calibration reliability check (brief's Step 5) shows a stark
  gap: transactions with a mean predicted score of ~53% are actually laundering only
  ~1.7% of the time. Across the whole test split, the mean predicted score is 7.31%
  against an actual prevalence of 0.177% — the model's average score is about 41x the
  true rate.
- **Root cause:** The direct, well-understood consequence of `scale_pos_weight=1324.94`
  (Phase 4) — the exact mechanism that lets the model rank rare positives above the
  overwhelming negative class inflates predicted scores for anything resembling a
  positive, and that inflation is what breaks calibration. It's the same knob doing two
  things: enabling useful ranking, and destroying probability meaning, as a package
  deal, not two separate problems.
- **The fix:** Not implemented, and deliberately scoped as future work rather than
  rushed in: Platt scaling or isotonic regression on the validation split would recover
  a calibrated probability if one were ever needed (e.g. showing an analyst a literal
  "X% chance of laundering" figure). Not needed for anything currently reported —
  precision@k, recall-per-typology, and the headline false-positive-reduction number
  all depend only on the model's *ranking*, which this distortion doesn't touch.
- **Interview angle:** Knowing that a metric result doesn't invalidate a model — because
  the thing that broke (calibration) isn't the thing the headline claims depend on
  (ranking) — is more useful than either ignoring the bad calibration plot or panicking
  about it. The brief asks for a calibration check specifically because "analysts triage
  by score," and reporting a genuinely broken calibration plot alongside a clear
  explanation of *why* it's broken and *why it doesn't matter for this project's actual
  claims* is a stronger, more honest result than a falsely reassuring plot would have
  been.

### The system flags transactions, not accounts — worth being precise about
- **The question:** `is_laundering` is a per-transaction label, and every metric in
  this project (precision@k, recall-per-typology, the headline reduction) is computed
  at transaction granularity. But nearly every feature driving those scores describes
  *account*-level rolling history and network position (`sender_out_7d_count`,
  `sender_graph_in_cycle`, ...), not the transaction's own standalone attributes. So
  which is it — is this an account-risk system or a transaction-risk system?
- **The answer:** Transaction-level classification, using account-level and
  network-level context features. Each row scored is one transaction; what makes that
  score meaningful is what the sender's and receiver's *recent history and network
  position* look like as of that transaction's timestamp, not the transaction's own
  amount and payment format alone (Feature 1-13 in `src/features.py`/
  `src/graph_features.py`'s module docstrings are all either sender- or
  receiver-account aggregates).
- **Checked, not assumed:** whether this matters in practice depends on how
  concentrated the alerts are on a small number of repeat accounts. At the headline
  operating point, the model's 10,011 alerted transactions touch 13,783 distinct
  accounts — 68.8% of the theoretical maximum (20,022, if every transaction's sender
  and receiver were entirely unique). The rules baseline's 297,564 alerts touch
  134,036 distinct accounts, and the 1,797 true laundering transactions in the test
  split touch 2,199 distinct accounts. None of these show heavy concentration on a
  small hub-account set — see `investigations/phase5_evaluation/
  03_transaction_vs_account_level_alerts.py`.
- **An honest gap this surfaces:** a real deployed system would very likely *bundle*
  multiple flagged transactions from the same account into one case for an analyst to
  review together, rather than handing over thousands of separate transaction-level
  alerts one at a time. That bundling/case-management step isn't built here — noted as
  a real scope gap, not something the headline 96.6% reduction number accounts for
  (it measures alert *volume*, and bundling would reduce both the model's and the
  baseline's effective review counts somewhat, though not necessarily by the same
  factor for each).
- **Interview angle:** "What's the unit of prediction, and does it match the unit a
  human actually acts on?" is a question worth being able to answer precisely for any
  ML system, not just this one — a fluent "yes we use account features" answer isn't
  the same as being able to say exactly what gets scored, what an analyst would
  actually review, and where the gap between the two currently sits.

### A caught arithmetic error: "43x" should have been "7.3x"
- **What happened:** While building the `investigations/` folder to make this
  project's ad-hoc analysis scripts reproducible (rather than leaving them in a local
  scratch directory), re-running the ACH-correlation investigation surfaced a
  discrepancy: ACH's laundering rate (0.7462%) divided by the dataset's overall
  prevalence (0.1019%) is **7.3x**, not the "~43x" figure already written into
  `reports/results.md`, `reports/challenges.md`, `PLAN.md`, and
  `portfolio/technical_overview.md` from Phase 4 — three of which were already merged
  to `main`.
- **Root cause:** A plain arithmetic slip made once, during Phase 4, that then
  propagated by being copied into four separate documents rather than recomputed each
  time — nothing caught it because nothing had re-derived the number from scratch
  since. The direction of the finding (ACH is heavily overrepresented in laundering
  transactions) was never in question; only the specific multiplier was wrong.
- **The fix:** Recomputed directly (4483 ACH-format laundering transactions / 600,797
  ACH-format transactions = 0.7462%; 5,177 / 5,078,345 overall = 0.1019%; ratio =
  7.32x) and corrected all four documents plus this investigation script's own
  docstring, which had been written with the same wrong figure before being checked
  against a live run.
- **Interview angle:** The catch itself is the point, not the mistake — a number that
  sat unchallenged in four merged documents got caught specifically *because* the
  investigation script was built to be re-run and re-verified, not just to reproduce a
  plot once and be trusted forever. This is the same argument for why this whole
  `investigations/` folder exists: a claim that was checked once and then copied
  around is a liability, and a script that recomputes it from the raw data every time
  it's run is what actually keeps a project's numbers honest over its lifetime.

---

## Phase 6 — Explainability

### SHAP's `expected_value` is silently wrong until you've already used the explainer
- **What happened:** The first faithfulness test written for `src/explain.py` asserted
  SHAP's additivity identity — `expected_value + sum(contributions)` must reconstruct
  the model's raw margin output — and it failed on *every* row, by a constant 0.0354 in
  log-odds. A constant offset across all rows rules out the obvious suspects (a
  misaligned feature matrix, a wrong column order, or a class-index mix-up would all
  produce row-varying errors), which pointed at the base value rather than the
  contributions.
- **Root cause:** On shap 0.45.1 + xgboost 2.0.3,
  `shap.TreeExplainer(model).expected_value` returns `array([logit(base_score)])`
  immediately after construction, and is **silently replaced** with a different, correct
  scalar during the first `shap_values()` call. The test read the attribute before
  explaining anything — the natural reading order — and so got the stale value. Verified
  precisely: pre-call it equals `logit(base_score)` to within floating point, post-call
  it equals the bias column of XGBoost's own `pred_contribs`, and only the post-call
  value satisfies additivity (max reconstruction error 1.9e-06 vs. 1.7e-02). Nothing
  warns, nothing raises, and the two values are close enough that a plotted waterfall
  would look entirely reasonable.
- **The fix:** Stop reading the attribute. `src/explain.shap_base_value` derives the
  base value as `margin - shap_values.sum(axis=1)`, which is immune to call ordering,
  and asserts the implied bias is constant across rows (raising if not) — so it verifies
  the property the code actually depends on instead of trusting a library attribute to
  mean what its name says. `tests/test_explain.py` pins the trap itself as a regression
  test, so a future shap/xgboost upgrade that changes this behaviour surfaces there
  rather than quietly shifting every reconstructed margin.
- **Scope, stated honestly:** no reported number was ever affected. Reason codes use
  only per-feature contributions and their ranking, and those were exact all along —
  verified bit-identical (max difference 0.00e+00) to XGBoost's native `pred_contribs`.
  This was a latent trap fixed before Phase 8/9 could build a SHAP waterfall or score
  decomposition on top of the wrong intercept, not a bug in any result.
- **Reproduction:**
  `investigations/phase6_explainability/01_shap_expected_value_lazy_initialisation.py`.
- **Interview angle:** The useful part is the diagnostic step, not the library quirk. A
  *constant* error and a *row-varying* error have different causes, and reading that
  distinction off the failure narrowed a 53-feature pipeline down to one scalar
  immediately. It's also an argument for testing the property rather than the plumbing:
  the additivity assertion existed only because "is this explanation faithful?" was
  written as an executable check, and that check is the sole reason a silent,
  plausible-looking offset was caught at all.

### A reason code faithful to SHAP leads with the same clause on 99.86% of alerts
- **What happened:** Phase 4 had already established that `payment_format_ACH` dominates
  this model's feature importance (86.6% of labelled laundering in HI-Small uses ACH, a
  laundering rate ~7.3x the dataset's overall prevalence). Under SHAP it dominates too,
  with mean |SHAP| of 1.75 versus 0.60 for the next feature. The Phase 6 question is
  what that does to the explanation layer specifically — and measured over the Phase 5
  alert queue (the 10,011 alerts at the equal-recall operating point, i.e. the queue an
  analyst would actually be handed), `payment_format_ACH` is the single largest positive
  contributor on **9,997 of 10,011 alerts (99.86%)**. Only 4 distinct features ever lead
  the sentence across the entire queue.
- **Why that's a real problem and not just an aesthetic one:** the reason code exists so
  an analyst can decide which case to open first. An opening clause identical on 99.86%
  of the queue carries essentially no discriminative information, however faithful it is
  to the model. "Flagged: the payment was made via ACH" is true, is what the model
  actually keyed on, and is useless for triage.
- **Root cause:** Not a bug in the explanation layer — it correctly reports a real
  property of the model, which in turn reflects a property of the dataset (and quite
  possibly of its synthetic generator, as Phase 4 already flagged). The mistake would
  have been to ship a single reason code and let this go unnoticed, or to quietly drop
  the feature to make the output look better.
- **The fix, and what was deliberately *not* done:** `build_reason_code` takes an
  `exclude_features` argument, and `PAYMENT_FORMAT_FEATURES` is passed to produce a
  second, **behavioural** variant alongside the faithful one. The exclusion applies to
  the *sentence only* — the payment-format one-hots remain in the model, so no reported
  metric changes and nothing is hidden. Both variants are reported because they answer
  different questions: *what is the model actually doing* (faithful) versus *what should
  an analyst look at on this case* (behavioural). The behavioural variant spreads across
  11 distinct leading features — `receiver_in_7d_distinct_counterparties` (41.0%),
  `amount_paid_usd` (27.8%), `sender_graph_fan_in_score` (15.5%),
  `receiver_in_1d_count` (12.5%) — i.e. the Phase 3 account/window and graph features
  this project exists to exploit.
- **A precision worth keeping:** it would be an overstatement to call the faithful
  variant information-free. Its second and third clauses still vary, so it produces
  7,098 distinct sentences across the 10,011-alert queue (behavioural: 8,844). The
  defensible claim is narrower, and is the one made in `results.md`: its *leading*
  clause, the part read first, is the same on ~99.9% of alerts.
- **Reproduction:**
  `investigations/phase6_explainability/03_ach_dominance_in_reason_codes.py`.
- **Interview angle:** This is the difference between "I generated SHAP explanations" and
  "I checked whether the explanations were usable." A global-importance bar chart shows
  ACH on top and looks like a finished result; only measuring the *distribution of
  leading contributors across the alert queue* reveals that the per-alert output had
  collapsed to a near-constant. It also lands on the right side of a real trade-off:
  faithfulness to the model and usefulness to the analyst are in tension here, and the
  honest resolution is to show both and disclose the gap, not to pick whichever presents
  better.

### Grounding a generated sentence is a testable property, so it gets tested
- **The question:** A reason code is generated text presented to an analyst as
  justification for escalating a case. The failure mode that matters is not a crash —
  it's a *fluent, confident sentence that misstates the transaction*: a clause paired
  with a different row's value, a sender/receiver mix-up, an off-by-one in the SHAP row
  alignment. Each produces output that reads perfectly and is false, and eyeballing five
  examples does not reliably catch any of them.
- **The answer:** Groundedness was made a checkable property rather than a matter of
  inspection. `describe_feature` is a pure, separately-testable value-to-words mapping,
  so `tests/test_explain.py` pins each feature family's rendering against a known raw
  value. On real data,
  `investigations/phase6_explainability/02_reason_code_spot_check_vs_raw_values.py`
  parses the number back out of each generated clause and asserts it round-trips to that
  row's value in the modelling table, then independently cross-checks the row against the
  Phase 3 feature table (agreement to a relative 2-6e-08, i.e. float32 epsilon —
  `prepare_feature_matrix` downcasts, so absolute gaps reach 1.45 on a $63M value and are
  precision artifacts, not mismatches) and confirms the `payment_format` string maps to
  the one-hot column that is actually set.
- **A check that was initially fake, and had to be fixed:** the first version of that
  script "verified" each clause by comparing `describe_feature(f, v)` to
  `describe_feature(f, v)` — a tautology that would pass no matter how wrong the wiring
  was. Worth recording precisely because it's the characteristic way a verification
  script fails: it ran, printed reassuring `OK` lines, and checked nothing. The real
  check parses the rendered sentence and compares the extracted number to the raw value,
  with tolerances matching each family's formatting (0.5 for whole-unit counts and USD,
  0.005 for the two-decimal ratio features) and direction-based checks for the binary
  features that render no number at all.
- **Two rules that came out of the same concern:** only features with *positive* SHAP are
  ever cited, since a negative contribution means the feature argued against the alert
  and citing it as a reason would be actively misleading; and binary features are
  rendered by direction, so a zero-valued one-hot carrying positive SHAP renders as "the
  payment was **not** made via X" rather than a clause implying the opposite. An unknown
  feature name degrades to `name = value` instead of raising — a degraded reason code is
  recoverable in production, an exception in the explanation layer takes down an
  otherwise healthy alert queue.
- **Why it matters:** the reason code is the part of this system a regulator or auditor
  would actually read, and "trust me, I looked at a few" is not an answer for generated
  text at scale. Making groundedness an assertion means "every clause states a checkable
  fact about this transaction" is verified on every run rather than asserted once.

---

## Phase 7 — Monitoring

### The largest drift signal in the monitor is generated by our own feature pipeline

**Symptom.** The daily score-PSI series is flat and stable for three days and then steps
up ~4x on a single day, with no ramp: 0.0707, 0.0685, 0.0783, then **0.2845** on
2022-09-09 and 0.2856 on 09-10. A step is a much more interesting shape than a slope —
gradual drift is a population slowly moving, a step means something switched.

**Root cause.** Nothing in the data switched. Phase 3 bounded each day's account graph to
a trailing `graph_features.lookback_days` (7) window with incremental edge pruning, to
stop an unbounded graph growing without limit (see the Phase 3 entry above). Edges are
pruned when their date is `< date - 7 days`. HI-Small starts 2022-09-01, so:

- day 2022-09-08 → cutoff 2022-09-01 → nothing older exists, nothing is pruned
- day 2022-09-09 → cutoff 2022-09-02 → **2022-09-01's edges are evicted, for the first time**

and 2022-09-01 is HI-Small's single largest day: 1,114,921 of 5,078,345 transactions,
22.0% of the dataset. So on 09-09 every account's graph features are recomputed on a graph
that has just lost 22% of the dataset's edges at once. The first eviction is also the
largest possible eviction, and it lands on exactly the day the step appears.

**Evidence.** 8 of the top 10 CSI drivers across the 09-08 → 09-09 boundary are
`*_graph_*` features (`sender_graph_in_degree` 1.155, `receiver_graph_in_degree` 1.138,
`receiver_graph_fan_out_score` 1.111, ...), not the velocity or amount families. In the
score distribution, the reference's bottom decile drains from 4.23% of the day's rows to
1.27% overnight, contributing 0.180 of the 0.285 total — those are precisely the
low-history, low-connectivity transactions that evicting the largest day of edges removes.
`investigations/phase7_monitoring/02_graph_lookback_eviction_step_change.py` asserts the
lookback arithmetic lands on 09-09 rather than eyeballing the coincidence.

**Why it matters.** This is the single most useful thing Phase 7 produced, and it is a
negative result. A production monitor would have paged someone to investigate a data
incident that does not exist. The generalisable lesson is that **a drift monitor watches
the features, not the world**, so any scheduled transformation inside the feature
pipeline — a lookback evicting its first window, a lookup table refreshing, a backfill
landing — shows up as drift indistinguishable from a real population change. Anything
with a schedule needs to be known to whoever reads the monitor. The Phase 3 decision
itself is still right; it is the interaction with monitoring that had to be discovered
rather than assumed.

### Most of the feature-level drift is the monitor watching its own windows fill up

**Symptom.** 23 of 54 features land in the "significant" CSI band (>0.25) between the
train and test splits, led by `receiver_in_30d_count` at CSI 3.08 (mean 8.00 → 25.55) and
`sender_out_30d_count` at 1.65 (mean 2,113.9 → 7,889.7). A model-risk function seeing 23
significant input shifts inside an 18-day window would pull the model.

**Root cause.** HI-Small spans ~17–18 days, but `config.yaml`'s `windows.rolling_days` is
`[1, 7, 30]`. A 30-day window is longer than the entire dataset, so `*_30d_count` cannot
do anything on day 3 except be smaller than it is on day 10 — every account's trailing
count is still accumulating toward a steady state it never reaches. The 7-day windows have
the same problem across the dataset's first week. That is **warm-up, not drift**: the
feature is converging, and the population behind it is unchanged. This was already known
as a *feature interpretation* caveat (it is in CLAUDE.md's Dataset section); what was not
anticipated is that it dominates the *monitoring* readout.

**The discriminating check.** If the population genuinely moved, features with no window
at all should move too. They do not. Splitting the CSI table by whether a feature depends
on a trailing window:

| Feature group | stable | moderate | significant |
|---|---|---|---|
| windowed | 14 | 10 | 22 |
| non-windowed | 7 | 0 | 1 |

`amount_paid_usd` — a per-transaction value with no window — is stable at CSI 0.0669,
against a windowed maximum of 3.0752. Per-day means confirm the mechanism directly: the
windowed counts climb from the dataset's very first day, which is what a filling window
looks like and not what a population shift looks like.
(`investigations/phase7_monitoring/01_window_warmup_vs_real_drift.py`.)

**Why it matters.** The honest reading of the CSI table is not "23 features have drifted";
it is "23 features are still warming up, and this dataset is too short to distinguish
warm-up from drift for any window longer than about a week." Reporting the raw count
without that decomposition would have been technically true and substantively misleading.

### A drift monitor that was structurally blind to binary features

**Symptom.** The first version of `characteristic_stability` reported CSI **0.0000** for
`payment_format_Reinvestment` on a train → test comparison in which that payment format
went from 15.8% of transactions to *exactly zero*. It reported 0.0000 for
`payment_format_ACH` too — the single most important feature in the model (Phase 4), the
one 86.6% of labelled laundering uses.

**Root cause.** Every quantile of a 0/1 column is the same value. `np.quantile` on a
binary feature returns the same number for the 10th through 90th percentile, the
de-duplicated edge array collapses to `[-inf, inf]`, and a single bin holds 100% of both
distributions by construction, so the PSI is 0.0 whatever happened. The `n_bins_effective`
column already exposed this (it read 1), but a monitor whose reassuring output requires a
second column to interpret is a monitor that will be misread.

**Fix.** Features with at most `max_categorical_cardinality` (10) distinct reference values
route through `category_share_psi`, which treats each distinct value as its own bin and
compares shares directly, over the *union* of reference and current categories so that a
brand-new category and a vanished one both contribute. `characteristic_stability` reports
which path each feature took in a `method` column, so the table stays auditable rather
than magic. Three previously-invisible binary features moved into the significant band
(20 → 23) and `payment_format_Reinvestment` went from a reported 0.0000 to rank 2 at
CSI 1.9169.

**What the fix then surfaced, which is a real finding.** All 481,056 Reinvestment
transactions in HI-Small fall on a single calendar day — 2022-09-01, where the format is
43.15% of transactions — and it never appears again. That is the only non-windowed feature
in the significant band, and unlike the warm-up drift it is genuine: the generator emitted
a payment format for one day and stopped.

**Why it matters.** The failure mode is the dangerous direction. A monitor that misses
drift reports *reassurance*, and reassurance is acted on. This one would have reported
"stable" for the vanishing of the model's top feature. It is also a good argument for
testing a metric against a case where you already know the answer: the bug was invisible
in every synthetic normal-distribution test and only appeared when the module was pointed
at the real one-hot columns.

### The one unambiguous population change is the one the monitor cannot see

**Symptom.** From 2022-09-11 onward HI-Small is not the same dataset. Daily volume
collapses from 654,467 rows to 396, 281, 184, 121, 46, 46, 23, 11. The laundering rate
goes from ~0.09% to 59.12%. Payment format becomes 100% ACH. Aggregated into one slice,
that tail's score PSI against the training distribution is **10.4044** — off any
conventional scale. The daily monitor reports `insufficient_data` for every single day of
it and never scores it.

**Root cause, and why it is not a bug.** Every one of those days is below
`monitoring.min_slice_size` (1,000), so each is correctly refused rather than scored on 46
rows of sampling noise. The floor is doing its job. Lowering it would not fix anything: it
would trade a blind spot for false alarms, since PSI over 10 quantile bins on 23 rows is
noise. Both the blind spot and the noise are real, and no choice of threshold removes both.

**The actual answer.** A distribution monitor needs a **volume monitor** beside it. Daily
row count falling from 654,467 to 11 is trivially detectable, needs no binning, no
reference distribution, and no minimum sample size. The mistake would be treating PSI as
*the* monitoring story rather than one signal among several — PSI answers "did the shape
of the population change", and it can only answer that when there is a population to
measure.

**The knock-on check that mattered more.** That tail sits inside the test split and holds
655 of its 1,797 positives — **36.4% of all test-split positives in 0.109% of its rows**.
That is a direct threat to the project's headline result, so the Phase 5 number was
re-derived with the tail removed, using the same `evaluate.py` functions rather than a
fresh calculation:

| Population | Baseline alerts | Baseline recall | Model alerts | FP reduction |
|---|---|---|---|---|
| Full test split | 297,564 | 68.8% | 10,011 | 96.6% |
| Body only | 296,597 | 61.9% | 11,140 | 96.2% |
| Tail only | 967 | 80.9% | 765 | 20.9% |

The headline holds — 0.4 percentage points of movement. The tail-only number is low for a
reason that is not a model failure: at ~59% prevalence there are almost no false positives
left to remove, so the rules baseline is already close to right and there is no headroom.
Worth stating plainly because "the model looks good because a third of the positives are
in a trivially-separable tail" is exactly the objection a sharp reviewer would raise, and
the answer is a number rather than a defence.
(`investigations/phase7_monitoring/03_tail_population_psi_cannot_see.py`.)

---

## Phase 8 — Demo artifact

### Shipping a *sample* to the app without letting it misrepresent the result

- **The question:** The deployed app can't hold the test split — 1,015,669 rows with 108
  feature/SHAP columns is far past Streamlit Community Cloud's ~1GB. So it gets a sample.
  But the project's entire claim is about an alert queue and a ranking, and a sample of a
  ranking is not a ranking. What exactly can be sampled without the deployed demo quietly
  saying something the pipeline never said?
- **The answer:** Separate the two things a sample can damage, and protect them
  differently.
  1. **Rank identity.** Compute scores, ranks and the equal-recall queue over the
     *complete* test split first, then sample rows. The shipped `rank` column is a true
     rank out of 1,015,669 (recorded as `ranks_are_out_of` in the metrics JSON). Had the
     sample been drawn first and ranked after, "rank 1" in the app would have meant
     "best of the 30,000 rows that survived sampling" — a claim nobody made, displayed
     as if the model made it.
  2. **Queue completeness.** The 10,011-alert queue *is* the headline result, so it ships
     whole and unsampled. Only negatives outside it are sampled, stratified across score
     deciles (a flat draw from this skew returns almost nothing above the threshold, and
     the app's "browse below the cut-off" view would have had no rows).

  Both are asserted in CI against the committed artifact, not just intended: the queue
  must be exactly ranks 1..N with no holes, and score must be monotone in rank.
- **Why it matters:** "We had to subsample for the demo" is a sentence that hides a range
  of sins, from harmless to disqualifying. The distinction between *sampling which rows
  you display* and *sampling before you compute the thing you're claiming* is the whole
  question, and being able to say which one you did — and point at a test that enforces
  it — is the difference between a demo and a misrepresentation.

### The tail had to be over-represented, so it got flagged instead of fixed

- **The question:** Phase 7 established that the 2022-09-11+ tail is a different
  population holding 36.4% of test positives in 0.109% of test rows. The demo sample keeps
  every positive (at 0.18% prevalence, a uniform 30,000-row draw would contain ~53
  laundering transactions — too few to show 8 typologies or let anyone click through real
  examples). Keeping every positive therefore drags the tail in at **3.097%** of the
  artifact versus **0.109%** of the real test split, roughly 28x its true weight.
- **The answer:** Don't correct it. Any reweighting that fixed the tail's share would have
  to drop positives or drop queue rows, damaging the two things the previous entry exists
  to protect. Instead: flag it per row (`is_tail_population`), record *both* shares in the
  metrics JSON, and re-derive the headline without the tail in the same run (96.2% vs.
  96.6%) so the artifact carries its own robustness check. A CI test asserts the sampled
  share exceeds the true share — i.e. it pins the direction of the distortion so it can't
  silently flip without someone noticing.
- **Why it matters:** The instinct is to make the sample look representative. But a sample
  built for *demonstrating a ranked queue* has different requirements than one built for
  estimating a population statistic, and quietly reweighting to look unbiased would have
  broken the former to fake the latter. Measuring and disclosing the distortion is both
  more honest and more useful than removing it.

### The behavioural reason code diverges only where the model is alerting

- **What broke:** A test asserting the faithful and behavioural reason-code variants
  differ on >90% of the artifact failed at **55.67%**. Phase 6 had measured that
  `payment_format_ACH` leads the faithful code on 99.86% of the alert queue, so ~99%
  divergence seemed like the obvious expectation.
- **Root cause:** The expectation was wrong; the artifact was right. Splitting divergence
  by queue membership shows **99.87% inside the queue** (independently reproducing Phase
  6's 99.86% on a differently constructed sample) and **33.53% outside** it. The alert
  queue is 99.9% ACH; the off-queue sample is 40.4% Cheque, 27.9% Credit Card and only
  14.8% ACH. For a non-ACH transaction the `payment_format_ACH` one-hot is 0 and its SHAP
  contribution is *negative* — mean −1.52 versus +2.25 on ACH rows, negative on 99.8% of
  non-ACH rows — meaning it argues against the alert. `top_contributors` is positive-only
  by design (Phase 6: a feature that argued against the alert is not a reason for it), so
  the one-hot never enters the sentence and excluding it changes nothing. 100% of the rows
  where the two variants are identical are non-ACH.
- **The fix:** Assert the property that's actually true and actually matters — >99%
  divergence *on the alert queue* — rather than a whole-artifact threshold that would have
  silently encoded the sampling mix into a test. Written up with the numbers in
  `investigations/phase8_demo_artifact/01_reason_code_divergence_is_queue_specific.py`.
- **Why it's a good interview story:** The failing test was the useful event, but not
  because it found a bug — there wasn't one. It found an assumption that had been carried
  forward from Phase 6 without noticing it was conditional on the alert queue. The
  temptation with a "close enough" failure like 56% vs. 90% is to relax the threshold
  until it passes; the right move was to work out which population the original 99.86%
  described and re-scope the assertion to it.

### `np.int64` does not survive `json.dump(default=str)` — it becomes a string

- **What broke:** Nothing, in the end — it was caught before the second full build run,
  but it's a trap worth recording because it fails *silently*. The metrics JSON is written
  with `json.dump(..., default=str)`, and DataFrame records went in via `to_dict`.
- **Root cause:** `np.float64` subclasses Python `float` and serialises as a number, so
  most of the file looked fine. `np.int64` does **not** subclass `int`, so it falls
  through to `default=str` and serialises as `"137"` — a quoted string. Every count in
  `recall_per_typology` would have reached the app as a string, with nothing raising
  anywhere; the failure would have surfaced as odd sorting or string concatenation in a
  chart, far from its cause. Separately, `NaN` from the thin PSI slices would have been
  written as the bare token `NaN`, which Python accepts and `JSON.parse` rejects.
- **The fix:** A single `records()` helper routing every DataFrame through pandas'
  `to_json` (which emits real JSON numbers and `null`), plus two tests: one asserting
  typology counts come back as `int` and not `str`, one asserting the file contains no
  `NaN` token.
- **Why it's a good interview story:** `default=str` is the standard "just make it
  serialise" reflex, and it converts a loud `TypeError` into silent data corruption in a
  file that crosses a process boundary. The general lesson is that a serialisation
  fallback whose job is to prevent crashes will, by construction, also prevent you from
  finding out that something was unserialisable.
