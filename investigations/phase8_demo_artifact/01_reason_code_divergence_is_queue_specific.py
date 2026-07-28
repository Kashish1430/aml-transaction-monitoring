"""Why the two reason-code variants disagree on ~56% of the demo artifact, not ~99%.

Supports the "the behavioural reason code diverges only where the model is alerting"
entry in reports/challenges.md's Phase 8 section.

Phase 6 established that `payment_format_ACH` leads the *faithful* reason code on 99.86%
of the 10,011-alert queue, which is why a behavioural variant (payment-format one-hots
excluded from the sentence) exists at all. The natural expectation when that variant was
carried into the Phase 8 demo artifact was that the two variants would differ on
essentially every row — a test asserting >90% divergence was written on that assumption
and failed at 55.67%.

The assumption was wrong, and the artifact is right. This script shows why, by splitting
the divergence by alert-queue membership and cross-tabulating against payment format.

Run from the repo root, after building the artifact:
    venve/python.exe investigations/phase8_demo_artifact/01_reason_code_divergence_is_queue_specific.py
"""

import numpy as np
import pandas as pd

ARTIFACT = "app/data/demo_alerts.parquet"

demo = pd.read_parquet(ARTIFACT)
faithful = demo["reason_code_faithful"].astype(str)
behavioural = demo["reason_code_behavioural"].astype(str)
differ = faithful != behavioural
in_queue = demo["in_alert_queue"].to_numpy()

print(f"Artifact: {len(demo):,} rows ({in_queue.sum():,} in the equal-recall alert queue)\n")

print("=== Reason-code divergence, split by alert-queue membership ===")
print(f"  overall            : {differ.mean():.2%}")
print(f"  inside the queue   : {differ[in_queue].mean():.2%}  (n={in_queue.sum():,})")
print(f"  outside the queue  : {differ[~in_queue].mean():.2%}  (n={(~in_queue).sum():,})")

ach_leads = faithful.str.startswith("Flagged: the payment was made via ACH")
print("\n=== How often the faithful code LEADS with the ACH clause ===")
print(f"  inside the queue   : {ach_leads[in_queue].mean():.2%}   <- Phase 6 reported 99.86%")
print(f"  outside the queue  : {ach_leads[~in_queue].mean():.2%}")

print("\n=== Payment-format mix explains the gap ===")
for label, mask in (("inside the queue", in_queue), ("outside the queue", ~in_queue)):
    shares = demo.loc[mask, "payment_format"].value_counts(normalize=True).head(4)
    print(f"  {label}:")
    for fmt, share in shares.items():
        print(f"      {fmt:<14} {share:.1%}")

# The mechanism: for a non-ACH transaction the one-hot is 0 and its SHAP contribution is
# negative (it argues AGAINST the alert). `top_contributors` is positive-only by design,
# so the feature never enters the sentence and excluding it changes nothing.
identical = ~differ
print("\n=== The mechanism ===")
non_ach_share = (demo.loc[identical, "payment_format"] != "ACH").mean()
print(f"  rows where the two variants are IDENTICAL : {identical.sum():,}")
print(f"  ...of which the payment format is NOT ACH : {non_ach_share:.1%}")

shap_ach = demo["shap_payment_format_ACH"].to_numpy()
is_ach = (demo["payment_format"] == "ACH").to_numpy()
print(f"\n  mean SHAP of payment_format_ACH when format IS  ACH: {shap_ach[is_ach].mean():+.4f}")
print(f"  mean SHAP of payment_format_ACH when format NOT ACH: {shap_ach[~is_ach].mean():+.4f}")
print(
    f"  share of non-ACH rows where that contribution is negative: "
    f"{(shap_ach[~is_ach] < 0).mean():.1%}"
)

assert non_ach_share > 0.99, "identical-variant rows should be overwhelmingly non-ACH"
assert np.mean(shap_ach[~is_ach] < 0) > 0.99, "the ACH one-hot should argue against non-ACH rows"

print(
    "\nConclusion: the variants are designed to diverge where the model is alerting, and\n"
    "they do (99.9% inside the queue). Outside it the population is mostly Cheque/Credit\n"
    "Card, the ACH one-hot contributes negatively, positive-only `top_contributors` drops\n"
    "it anyway, and both variants produce the same sentence. The ~56% overall figure is a\n"
    "property of the artifact's sampling mix, not a defect — so the test asserts >99% on\n"
    "the queue rather than a whole-artifact threshold that would encode the sample's shape."
)
