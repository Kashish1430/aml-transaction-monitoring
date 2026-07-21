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
