import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Transaction Monitor', layout='wide')

# The five raw inputs every transaction needs
INPUTS = ['amount', 'customer_typical_amount', 'transactions_last_hour',
          'minutes_since_previous', 'account_age_days']
LABELS = {'amount': 'Amount ($)', 'customer_typical_amount': "Customer's typical amount ($)",
          'transactions_last_hour': 'Transactions in the last hour',
          'minutes_since_previous': 'Minutes since previous transaction',
          'account_age_days': 'Account age (days)'}

# ---- load the trained model (made in the Colab notebook, unchanged) ----
bundle = joblib.load('model.joblib')
model = bundle['model']
history_scores = bundle['history_scores']
features = bundle['features']


# ---- PROTECTION 2: the range of each input in the training data ----
# The model only learned from history.csv. Outside these ranges its score has nothing to lean on.
@st.cache_data
def training_ranges():
    try:
        history = pd.read_csv('history.csv')
        return {c: (float(history[c].min()), float(history[c].max())) for c in INPUTS}
    except Exception:
        return None


RANGES = training_ranges()


def score(amount, typical, last_hour, gap, age):
    """Turn one transaction into the five features and return its anomaly score."""
    row = pd.DataFrame([{
        'log_amount': np.log1p(amount),
        'amount_ratio': amount / typical,
        'transactions_last_hour': last_hour,
        'log_gap': np.log1p(gap),
        'log_age': np.log1p(age),
    }])[features]
    return -model.score_samples(row)[0]


# ---- PROTECTION 1: check the data before scoring ----
def data_problems(values):
    """Return a list of problems with one transaction's raw inputs (empty list = valid)."""
    problems = []
    for c in INPUTS:
        v = pd.to_numeric(values.get(c), errors='coerce')
        if pd.isna(v) or np.isinf(v):
            problems.append(f'{c} is missing or not a number')
        elif c in ('amount', 'customer_typical_amount') and v <= 0:
            problems.append(f'{c} must be above 0 (got {v:g})')
        elif v < 0:
            problems.append(f'{c} cannot be negative (got {v:g})')
    return problems


def outside_range(values):
    """Return the inputs that fall outside what the model saw in training."""
    if RANGES is None:
        return []
    notes = []
    for c in INPUTS:
        low, high = RANGES[c]
        v = float(values[c])
        if v < low or v > high:
            notes.append(f'{LABELS[c]} = {v:,.2f} (past data: {low:,.2f} to {high:,.2f})')
    return notes


# ---- sidebar: the review budget ----
st.title('Transaction Monitor')
st.write('Each new transaction is scored by Isolation Forest. '
         'If the score is above the cutoff, it goes to review.')

pct = st.sidebar.slider('Send this share of transactions to review (%)', 1, 10, 4)
cutoff = np.percentile(history_scores, 100 - pct)
st.sidebar.write(f'Score cutoff: **{cutoff:.3f}**')
st.sidebar.caption('The cutoff is set from past transactions: '
                   'a transaction scoring above it is more unusual than the top '
                   f'{pct}% of history.')
if RANGES is None:
    st.sidebar.warning('history.csv is missing, so the app cannot check whether a transaction is '
                       'inside the range of past data. Scores still work, but treat unusual '
                       'values with extra care and restore history.csv to the repository.')

tab1, tab2 = st.tabs(['Check one transaction', 'Live stream'])

# ---- tab 1: type in one transaction ----
with tab1:
    c1, c2 = st.columns(2)
    amount = c1.number_input('Amount ($)', min_value=1.0, value=60.0)
    typical = c1.number_input("Customer's typical amount ($)", min_value=1.0, value=50.0)
    last_hour = c1.number_input('Transactions in the last hour', min_value=0, value=1)
    gap = c2.number_input('Minutes since previous transaction', min_value=0.0, value=120.0)
    age = c2.number_input('Account age (days)', min_value=0, value=700)

    if st.button('Score this transaction'):
        values = dict(zip(INPUTS, [amount, typical, last_hour, gap, age]))
        problems = data_problems(values)
        if problems:
            st.error('NOT CHECKED: ' + '; '.join(problems) + '. Correct the values and try again.')
        else:
            s = score(amount, typical, last_hour, gap, age)
            unusual_inputs = outside_range(values)
            if s > cutoff:
                st.error(f'REVIEW  (score {s:.3f}, cutoff {cutoff:.3f})')
            elif unusual_inputs:
                # Do not show a green OK the model cannot back up
                st.warning(f'CANNOT CONFIRM: send to manual review  (score {s:.3f}, cutoff {cutoff:.3f})')
            else:
                st.success(f'OK  (score {s:.3f}, cutoff {cutoff:.3f})')
            if unusual_inputs:
                st.warning('This transaction is outside the range of the past data the model learned from:\n\n- '
                           + '\n- '.join(unusual_inputs)
                           + '\n\nThe model has never seen values like these, so its score is not reliable '
                             'here. A low score does **not** mean the transaction is safe. '
                             'Have a reviewer check it by hand.')
            share = (history_scores < s).mean() * 100
            st.write(f'This is more unusual than {share:.0f}% of past transactions.')

# ---- tab 2: transactions arriving one by one ----
with tab2:
    st.write('Press the button. The app receives the new transactions one at a time '
             'and scores each one as it arrives.')
    source = st.radio('Transactions', ['Demo stream (new_transactions.csv)', 'Upload a CSV'], horizontal=True)
    upload = None
    if source == 'Upload a CSV':
        upload = st.file_uploader('CSV with columns: transaction_id, ' + ', '.join(INPUTS)
                                  + ' (needs_review is optional)', type='csv')
    speed = st.select_slider('Speed', options=['Slow', 'Medium', 'Fast'], value='Medium')
    pause = {'Slow': 0.5, 'Medium': 0.15, 'Fast': 0.02}[speed]

    if st.button('Start the stream'):
        # PROTECTION 1: make sure the file can be read and has every column the model needs
        try:
            if source == 'Upload a CSV':
                if upload is None:
                    st.error('No file uploaded yet. Choose a CSV above, then press Start the stream.')
                    st.stop()
                stream = pd.read_csv(upload)
            else:
                stream = pd.read_csv('new_transactions.csv')
        except Exception as e:
            st.error(f'The transaction file could not be read ({type(e).__name__}). Nothing was scored. '
                     'Check that it is a valid CSV file, then try again.')
            st.stop()

        missing_cols = [c for c in ['transaction_id'] + INPUTS if c not in stream.columns]
        if missing_cols:
            st.error('Nothing was scored: the file is missing these columns: **' + ', '.join(missing_cols)
                     + '**. Without them the model cannot score any transaction. Fix the file '
                       '(or ask whoever exports it to include these columns) and run the stream again.')
            st.stop()

        box = st.empty()
        counts = st.empty()
        st.subheader('Sent to review')
        table = st.empty()
        st.subheader('Could not be checked (data problem)')
        bad_table = st.empty()
        flagged, not_checked = [], []
        score_flags = out_of_range_count = 0

        for i, r in stream.iterrows():
            values = {c: r[c] for c in INPUTS}
            problems = data_problems(values)

            if problems:
                # Do not guess and do not skip silently: record it so a person can follow up
                not_checked.append({'id': r['transaction_id'], 'problem': '; '.join(problems)})
                box.warning(f"{r['transaction_id']}: DATA PROBLEM ({'; '.join(problems)})  ->  NOT CHECKED")
            else:
                values = {c: float(values[c]) for c in INPUTS}
                s = score(values['amount'], values['customer_typical_amount'],
                          values['transactions_last_hour'], values['minutes_since_previous'],
                          values['account_age_days'])
                unusual_inputs = outside_range(values)
                out_of_range_count += bool(unusual_inputs)
                msg = f"{r['transaction_id']}: ${values['amount']:.2f}  score {s:.3f}"
                if s > cutoff or unusual_inputs:
                    score_flags += s > cutoff
                    reason = 'high score' if s > cutoff else 'outside past data range'
                    flagged.append({'id': r['transaction_id'], 'amount': values['amount'],
                                    'typical': values['customer_typical_amount'],
                                    'last_hour': values['transactions_last_hour'],
                                    'score': round(s, 3), 'reason': reason})
                    box.error(msg + f'  ->  REVIEW ({reason})')
                else:
                    box.info(msg + '  ->  ok')

            counts.write(f'Received **{i + 1}** of {len(stream)}.  '
                         f'Sent to review: **{len(flagged)}**.  '
                         f'Could not be checked: **{len(not_checked)}**')
            if flagged:
                table.dataframe(pd.DataFrame(flagged), hide_index=True)
            if not_checked:
                bad_table.dataframe(pd.DataFrame(not_checked), hide_index=True)
            time.sleep(pause)

        st.success(f'Done. {len(flagged)} transactions were sent to review.')

        if not_checked:
            st.error(f'**{len(not_checked)} transactions were NOT checked** because their data was missing '
                     'or invalid (listed above). They have not been cleared. Review them by hand now, and '
                     'ask the data team to fix the source, then run them through the monitor again.')
        if out_of_range_count:
            st.warning(f'{out_of_range_count} transactions had values outside the range of the past data. '
                       'They were sent to review because the model cannot score them reliably.')

        # PROTECTION 2: does this batch still look like the past data?
        scored = len(stream) - len(not_checked)
        if scored >= 100:
            rate = score_flags / scored * 100
            if rate > 2 * pct or rate < pct / 2 or out_of_range_count / scored > 0.05:
                st.warning(f'**This batch does not look like the past data.** The score flagged {rate:.1f}% of '
                           f'transactions, but the cutoff was set to flag about {pct}%. '
                           'Customer behavior may have changed, so the cutoff and scores may no longer mean what '
                           'they used to. Keep reviewing the flagged list, but tell the analytics team: the model '
                           'should be checked against recent data before the next batch.')

        # if the file has an answer key, check how we did
        if 'needs_review' in stream.columns:
            answer = stream.set_index('transaction_id')['needs_review']
            hits = sum(answer[f['id']] for f in flagged)
            st.write(f'{hits} of the {len(flagged)} flagged transactions really needed '
                     f'review. In total, {answer.sum()} transactions needed review.')
