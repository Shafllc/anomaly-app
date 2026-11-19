# Transaction Monitor: safer version

A copy of the class Anomaly Detection App (Isolation Forest), with two protections added. **The trained model (`model.joblib`) is unchanged.**

**Live app:** https://anomaly-app-safer.streamlit.app

## What changed

**Protection 1: missing or invalid transaction data**
- The original scored a transaction with a missing typical amount as if it were normal (a $2,250 purchase passed as "ok"), and crashed on a typical amount of $0, losing the whole review queue.
- Now every transaction is checked before scoring. A file missing required columns is stopped with a message naming them. A row with missing, non-numeric, zero, or negative values is **not scored and not cleared**: it goes to a "Could not be checked" list with the reason, and the stream keeps going.

**Protection 2: results the model cannot back up**
- The model only learned from the ranges in `history.csv`. The original showed a green "OK" for a $50,000 purchase (past maximum: $5,445) and for an account 55 years old.
- Now a transaction outside the past data range is never shown as "OK". It shows "CANNOT CONFIRM: send to manual review" with the values that are out of range. In the stream, these are sent to review.
- After each stream, the app compares the share flagged with the review budget. If it is far off (more than double or less than half), it warns that customer behavior may have changed and the model should be checked.

## Files

- `app.py`: the app (protections marked `PROTECTION 1` and `PROTECTION 2` in the code)
- `model.joblib`: the trained model from the class notebook, unchanged
- `history.csv`: past transactions (used for the cutoff and the range check)
- `new_transactions.csv`: the demo stream
- `requirements.txt`
- `test_data/`: synthetic files that trigger each failure. In the app, choose **Upload a CSV** on the Live stream tab.
  - `test_bad_data.csv`: 60 demo rows with 5 broken rows (missing values, a $0 typical amount, text instead of a number, a negative time)
  - `test_missing_column.csv`: an export without `customer_typical_amount`
  - `test_out_of_range.csv`: 60 demo rows plus 4 transactions outside the past data range
  - `test_shifted_behavior.csv`: the demo stream after a change in customer behavior (more, faster transactions)
