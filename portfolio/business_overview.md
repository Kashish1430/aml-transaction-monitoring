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

### Stage 4 onward — not built yet
The system that actually turns this information into a ranked, reviewable alert list
(the headline "X% fewer alerts reviewed at the same catch rate" result), a plain-English
explanation for every flagged alert (so an analyst — and a regulator — can see *why* it
was flagged, not just trust a black box), a way to detect if the system's assumptions
have gone stale over time, and a live, click-through demo. These will be added to this
document as they're completed — see `PLAN.md` for the full build order.

## An honest caveat

This project uses a fully **synthetic**, publicly available dataset (IBM's AML
benchmark data), not real bank transactions — no institution's actual customer data is
involved. The methodology (how the rules baseline works, why the evaluation approach
was chosen, how the architecture mirrors real regulatory timelines) is built to reflect
real industry practice, and is discussed in more depth in `portfolio/technical_overview.md`
and `reports/challenges.md`, but the specific numbers in this project should be read as
a demonstration of approach, not a claim about performance on real-world data.

## Current status

Phases 0-3 of 11 are complete (data validation, rules baseline, feature engineering).
Next: building the actual predictive model (Phase 4) and measuring the headline
false-positive-reduction result against the Stage 2 baseline above (Phase 5).
