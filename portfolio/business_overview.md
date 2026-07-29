# AML Alert-Triage System — For Business & Non-Technical Readers

*A living document, updated as each build phase completes — see the "Current status"
section at the bottom for what's real today vs. what's still coming. Every number here
is traceable to `reports/results.md` or `reports/challenges.md`; nothing is estimated
for effect.*

## The one-paragraph version

Banks are legally required to monitor every transaction for signs of money laundering.
The standard tool for this — a rules engine ("flag anything over $10,000," "flag rapid
in-and-out transfers") — works, but it's blunt: it floods compliance teams with alerts,
the overwhelming majority of which turn out to be nothing. This project builds a system
that sits *after* that rules engine and re-ranks its alerts, so an analyst spends their
limited time on the handful that are actually worth investigating, without missing real
cases. It doesn't replace the bank's existing controls — it makes the human review step
that follows them dramatically more efficient.

## Why this is a real business problem, not just a modelling exercise

Every alert a rules engine raises has to be looked at by a trained analyst — that's a
regulatory requirement, not optional. When the vast majority of alerts are false alarms,
the bank is paying skilled compliance staff to repeatedly conclude "this was nothing,"
which is expensive, slow, and — because attention is a limited resource — makes it
*easier* for a genuinely suspicious case to get rubber-stamped through a fatigued
review process. Reducing the false-alarm volume without missing real cases is directly
a cost-and-risk problem for the bank, not just a data science exercise.

## What "money laundering" actually looks like (in plain terms)

Criminals don't launder money in one big obvious transaction — that would be caught
immediately. Instead, the patterns this project is built to catch look like:

- **Structuring ("smurfing"):** splitting a large sum into many smaller transfers, each
  just under the amount that would trigger automatic reporting (in the US, that
  threshold is $10,000) — the financial equivalent of sneaking a large amount through
  security one small piece at a time.
- **Fan-out:** one account suddenly sending money to many different, unrelated
  recipients — money deliberately scattered wide to make it harder to trace.
- **Fan-in:** the mirror image — many different accounts all sending money into one
  collection point.
- **Layering / pass-through:** money arriving in an account and leaving again almost
  immediately, rather than sitting there like a normal balance — the account is being
  used as a waypoint, not a destination.
- **Cycles:** money that eventually loops back toward where it started, after passing
  through several other accounts — designed to make the trail look like ordinary
  commerce rather than one person moving their own money around.

None of these are visible by looking at a single transaction in isolation — they only
show up when you look at an account's *recent history* and its *position in the wider
network of who-pays-whom*. That's the core insight the whole project is built around.

## What's been built so far, and what it means for the business case

### Stage 1 — Understanding how rare the real signal actually is
Before building anything, the project measured how often laundering actually occurs in
the data: **about 1 in every 1,000 transactions** (0.10%) — far rarer than the ~2%
originally assumed going in. This matters commercially: a system this rare a signal
can't be judged by "percent correct" (a system that never flags anything would already
be 99.9% "accurate" and completely useless). Every result later in this project is
measured in terms a compliance team actually cares about — how many real cases would be
caught for a given amount of analyst time spent, not a headline accuracy number that
would be meaningless at this scale.

### Stage 2 — Measuring the cost of today's status quo
The project then built a simple rules-based system representative of what many
institutions run today (flag large transactions, flag rapid in-and-out transfers, flag
suspiciously structured amounts). Result: **it flags 36 out of every 100 transactions**,
but only **about 1 in 600 of those flags is a real laundering case** (0.17% precision).
This is the baseline cost this project exists to reduce — every later stage is measured
against "did we shrink this haystack without losing the real cases inside it."

### Stage 3 — Giving the system the same context a human investigator would use
A human investigator doesn't judge one transfer in isolation — they ask "has this
account been unusually active lately?", "who else is this account connected to?", "does
money seem to just pass through this account rather than stay?" This stage built exactly
that context into the system: rolling history for every account (how much have they
sent/received in the last day/week/month, to how many different people) and their
position in the account network (do they sit at the center of a lot of money flowing
in from many sources, or fanning out to many destinations, or on a loop that returns to
itself). A first check already shows the expected direction — accounts later confirmed
as laundering show visibly more of these warning signs on average than ordinary accounts
— before any actual decision-making model has even been built yet.

### Stage 4 — building the system that actually ranks alerts
This stage trained the model that turns everything built so far into a single risk
score per transaction — the first version of the system that could hand an analyst a
ranked list instead of a flat pile of equally-weighted alerts. Early read: **looking
only at the top 100 highest-scored transactions out of over a million, 92 of them are
genuinely worth investigating** — compared to roughly 1 in 600 for the rules-based
status quo from Stage 2. That's not yet the project's final, audited headline number
(that comes once the model is measured properly against the Stage 2 baseline in the
next stage), but it's an early, honest signal the approach works.

Two things surfaced during this stage that are worth being upfront about, because
glossing over them would undercut the project's credibility, not protect it:
- A large share of the model's current performance rides on *one* strong signal (the
  payment channel used — ACH transfers account for the overwhelming majority of
  labelled laundering cases in this dataset). That could be a genuine real-world pattern
  or a quirk of how this training data was synthetically generated — this project can't
  fully tell which from the data alone, so it's disclosed as a caveat rather than
  quietly relied on. Tested by retraining without that one signal: performance drops
  substantially, but the model still comfortably beats the rules-based status quo, and
  the signals it leans on instead are exactly the account-history and network features
  built in the previous stage — a good sign those are carrying real weight, not just
  theoretical value.
- A second, "no rules, just watch for anything unusual" system was also built and
  tested alongside the main one, as a check for laundering patterns the labelled
  training data might not cover. It didn't work well as built — it mostly noticed a
  handful of extremely high-volume accounts (most likely large legitimate businesses),
  not laundering behavior — and that failure was investigated and understood rather
  than swept aside. Knowing *why* an approach didn't work, specifically, is worth more
  than a lucky number that happens to look good.

### Stage 5 — the headline result

This is the number the whole project has been building toward. Measured properly, on
transactions the system hadn't seen during training, against the Stage 2 rules-based
status quo re-measured on that exact same set of transactions (not a different, easier
comparison — the same population, so the two are actually comparable):

**At the same catch rate as the existing rules-based approach (68.8% of real laundering
cases caught), this system requires reviewing 96.6% fewer alerts to get there — 10,011
alerts instead of 297,564, on the same set of transactions.** In practical terms: a
compliance team using the existing rules-based approach on this population would need
to review roughly 1 in 3 of every transaction that passes through the bank; using this
system's ranking at the same catch rate, they'd review roughly 1 in 100 — without
missing any more real cases than they already were.

Breaking that catch rate down by the specific laundering pattern involved (structuring,
fan-in, fan-out, cycles, and four other recognized schemes), no single pattern is
dramatically weaker than the others — the system doesn't have an obvious blind spot
where an entire category of laundering slips through disproportionately.

One number worth being upfront about rather than glossing over: the system's internal
confidence scores don't translate directly into "this transaction has a 70% chance of
being laundering" — they're a reliable *ranking* (higher score genuinely means more
suspicious, relative to other transactions), but not a literal probability, because of
a design choice made to handle how rare real laundering is (see
`portfolio/technical_overview.md` for the technical detail). This doesn't affect the
headline result above, which is based entirely on ranking, but it would matter if a
future version of this system ever displayed a literal percentage-chance figure to an
analyst rather than a ranked queue.

### Stage 6 — telling the analyst *why*, in plain English

A ranked queue is only half of what a compliance analyst needs. The other half is a
reason: money laundering is a regulated area, so an analyst has to be able to justify
escalating a case, and an auditor has to be able to follow that justification months
later. "The model scored it 0.97" is not a justification.

So every alert now carries a sentence built from what actually drove its score — not a
generic template, but that specific transaction's own numbers. For example:

> *Flagged: the receiver received from 11 distinct counterparties in the past 7 days; a
> $18,756 payment; the receiver received from 10 distinct accounts in the graph window.*

Two things about this were treated as requirements rather than nice-to-haves:

- **The sentence only cites evidence that argued *for* the alert.** Some characteristics
  of a transaction make it look *more* legitimate; including those in a justification
  would actively mislead the person reading it.
- **Every claim in the sentence is checked against the transaction's real data,
  automatically, every time the pipeline runs.** Generated text that reads fluently but
  quotes the wrong figure is the genuine risk here — far more dangerous than an obvious
  error, because nobody catches it. So "does each sentence state the true numbers for
  this transaction?" is an automated test, not something spot-checked once by eye.

**A finding worth being upfront about.** When the system explains itself completely
faithfully, it opens with the same reason on 99.9% of alerts: the payment method used
(ACH — a common US bank transfer). That is genuinely what the model relies on most, and
it reflects a real pattern in this dataset, where the overwhelming majority of known
laundering used that one payment method. But a reason that is identical on virtually
every alert tells an analyst nothing about which case to open first. Rather than hide
that, the system produces **two** explanations per alert: the fully faithful one, and a
"behavioural" one that sets the payment method aside and describes the *account
behaviour* instead — how many counterparties, how much money, what the network structure
around the account looks like. The second is what's useful for triage; the first is what
keeps the system honest about what it's actually doing. Both are reported, and the gap
between them is documented rather than smoothed over.

### Stage 7 — noticing when the world stops matching the model

A model that was right in September is not automatically right in December. Customer
behaviour shifts, a bank launches a new payment product, a data feed breaks. The
uncomfortable part in AML specifically is that you **cannot wait to be told**: confirmed
laundering outcomes arrive weeks to months later, so if you wait for results to prove the
model degraded, you have already spent a quarter making bad decisions.

So this stage added an early-warning layer that watches the *shape of the data* rather
than the results. Every day, it asks: do the transactions coming in still look like the
ones the model learned from? It needs no outcomes, so it can raise a hand immediately.

The genuinely useful finding here is a cautionary one, and it is the reason this stage
earns its place rather than just producing a green dashboard. **The alarm fired, loudly,
and almost all of it was our own doing.** The system flagged a large, sudden shift on one
specific day. Investigating it properly rather than accepting it showed the data had not
changed at all — the model's own feature-building step had reached a scheduled point where
it starts "forgetting" transactions older than a week, and the day it first forgot happened
to be the busiest day in the dataset. Similarly, most of the flagged "changes" in
individual data points turned out to be measurements that were still warming up, because
some of them summarise a 30-day history in a dataset that only spans 18 days.

Three lessons that transfer directly to a real deployment:

- **A drift alarm watches the model's inputs, not the world.** Anything scheduled inside
  the data pipeline looks exactly like a genuine change in customer behaviour. Whoever
  reads the alerts has to know the pipeline's own calendar, or they will chase incidents
  that do not exist.
- **A monitor that misses something is more dangerous than one that over-fires**, because
  it reports reassurance and reassurance gets acted on. A real defect was found and fixed
  here: the first version was mathematically incapable of noticing a simple yes/no field
  changing, and it was silently reporting "all stable" for the single most influential
  input in the entire model. It now catches it.
- **Not every problem is a statistics problem.** The one unambiguous, dramatic change in
  this dataset — the last several days look nothing like the rest — is one the statistical
  monitor structurally cannot see, because those days contain too few transactions to
  measure anything reliably. The right answer is not a cleverer statistic; it is to also
  watch something trivially simple, like how many transactions arrived today.

This stage also produced a **direct stress-test of the headline result**. Since that
unusual tail of the data sits inside the period the model was scored on, the headline was
recalculated with it removed entirely. The 96.6% reduction becomes 96.2% — the result does
not depend on it. That check was run because a sceptical reviewer would rightly ask, and
the answer should be a number rather than a reassurance.

### Stage 8 — packaging the results so a reviewer can actually click through them

A working demo is worth more than any table of numbers, but the demo has to be honest
about what it's showing. The full dataset is five million transactions; a free hosted web
app can't hold that, so the app is given a prepared extract of 30,000 transactions
(15 MB) instead of the raw data.

The care here went into what an extract is allowed to change. Two things were protected:

- **The queue positions are real.** The model's scoring and ranking are done across the
  whole held-out period of one million transactions *first*, and only then are rows
  selected for the demo. So "alert #1" in the dashboard is genuinely the single
  highest-priority transaction the model found in that period — not merely the best one
  that happened to survive the extract. It would have been very easy to do this the other
  way round and show a number that quietly meant something weaker.
- **The alert queue is complete.** All 10,011 alerts behind the headline result are
  included, not a sample of them, because those alerts *are* the result.

The extract also inherits one distortion that couldn't be removed without breaking the
above: because every known laundering case is kept, the unusual final few days of the
dataset (described in Stage 7) end up making up about 3% of the demo data instead of their
true 0.1%. Rather than quietly rebalance it, every row is tagged so it can be filtered, and
both figures are recorded alongside the results — together with the headline recalculated
without those days at all (96.2% instead of 96.6%), so a reviewer can see for themselves
that the result doesn't rest on them.

Finally, the plain-English reason for each alert, and the full breakdown behind it, are
calculated in advance and stored with each row — so the dashboard can explain any alert
instantly without doing heavy computation while someone waits.

### Stage 9 — the dashboard an analyst would actually sit in front of

The system now has a working front end, built around what the job actually involves rather
than around the model. Four views:

- **Alert Queue** — the ranked worklist. Highest-priority alert first, with the
  plain-English reason attached to each row, and filters for laundering type, payment
  method, date, amount and specific accounts or banks. An analyst can export what they're
  looking at.
- **Alert Detail** — one alert, fully opened up: the reason for it, a chart showing exactly
  which factors pushed the score up and which pushed it down and by how much, and the
  complete underlying figures for anyone who wants to check the working.
- **Model Performance** — the honest scorecard: the headline comparison against the
  rules-based baseline, how well the model finds each of the eight laundering patterns, and
  the drift monitoring from Stage 7.
- **About** — what the system is, how it works, and every caveat, including that the data
  is synthetic and that the browsable table is a prepared sample.

Two things were deliberate. The dashboard **never re-orders the queue itself** — the
priority ranking was calculated across the full held-out period and is displayed exactly as
calculated. And wherever a view drops below the alert cut-off into sampled data, it *says
so on screen*, rather than presenting a sample count as though it were a real one.

Testing this found two faults that would otherwise have shown up on the public site: one
where clearing a filter crashed the queue view, and one where the app would have failed to
start at all once deployed, for a reason no amount of local testing would have revealed
(the test setup happened to hide it). Both are documented.

### Stage 10 — live, and open to anyone

The dashboard is deployed and publicly reachable at **https://aml-transaction-monitoring-839p8dzdafalfnkbpgwrpp.streamlit.app/** — no login, no install,
nothing to run. It costs nothing to host, which was a constraint set at the start of the
project rather than a happy accident.

### Stage 11 — the written summary

The project's front page now opens with the result and the live link, then explains the
business problem, why the system is built as an overnight batch process rather than a
real-time check, why network structure matters, why accuracy is the wrong measure here, and
every caveat worth knowing — including that the data is synthetic and that one payment-type
feature carries more of the result than is comfortable.

One small thing worth mentioning, because it reflects how the whole project was built: the
claim "every number in this document is traceable" is itself checked automatically. The
headline figures in the front page are compared against the model's own output files every
time the tests run, so if the pipeline is ever re-run and a number changes, the tests fail
until the write-up is corrected. It caught its first error within minutes of being written —
the page claimed 147 tests when there were 157.

### What's optional from here
Two extensions are planned but not built: a network view showing the web of accounts around
an alert, and a demonstration that the system could score transactions as they arrive. Both
are described in `PLAN.md`. Neither is needed for the project to stand on its own.

## An honest caveat

This project uses a fully **synthetic**, publicly available dataset (IBM's AML
benchmark data), not real bank transactions — no institution's actual customer data is
involved. The methodology (how the rules baseline works, why the evaluation approach
was chosen, how the architecture mirrors real regulatory timelines) is built to reflect
real industry practice, and is discussed in more depth in `portfolio/technical_overview.md`
and `reports/challenges.md`, but the specific numbers in this project should be read as
a demonstration of approach, not a claim about performance on real-world data.

## Current status

**Complete.** All 11 phases are done and the dashboard is live at
https://aml-transaction-monitoring-839p8dzdafalfnkbpgwrpp.streamlit.app/ — data validation, rules baseline, feature engineering, a trained model, the headline
result above, a plain-English justification on every alert, drift monitoring, the prepared
demo extract, the dashboard, deployment, and the written summary. Two optional extensions
(a network view, and live transaction scoring) are planned but not built.
