import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn.hmm import GMMHMM
import math
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
import ta
from sklearn.preprocessing import StandardScaler
from scipy.stats import percentileofscore
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.data.preprocessing import train_df, test_df, spy_df

#### Makes second subplot
fig2, (ax5, ax6, ax7, ax8, ax9, ax13, ax14, ax15) = plt.subplots(
    8, 1,
    figsize=(11, 10),
    sharex=True,
    gridspec_kw={'hspace': 0.7} 
)

fig2.suptitle(f"bull/bear market regimes for {stock}\nusing gaussian density hmm (model 2)", fontsize=14, fontweight='bold')

ax8.set_xlabel("time")
ax8.plot(test_df.index[split_data:], test_df['price vs ma50'][split_data:], label='price vs ma50')
ax8.set_title("price vs ma50")

ax14.set_xlabel("time")
ax14.plot(test_df.index[split_data:], test_df['beta'][split_data:], label='beta')
ax14.set_title("beta")

ax9.plot(test_df.index[split_data:], test_df['volatility_rolling'][split_data:], label='volatility_rolling')
vol_mean = test_df['volatility_rolling'].mean()
ax9.axhline(vol_mean)
ax9.set_xlabel("time")
ax9.set_ylabel("volatility_rolling")
ax9.set_title(f"{stock} vol")
ax9.legend()

ax15.plot(test_df.index[split_data:], test_df['vol_of_vol'][split_data:], label='vol_of_vol')
ax15.set_xlabel("time")
ax15.set_ylabel("vol_of_vol")
ax15.set_title(f"{stock} vol of vol")
ax15.legend()

ax6.set_xlabel("time")
ax6.set_ylabel("price")
ax6.plot(test_df.index[split_data:], test_df['Close'][split_data:], label='close')
ax6.legend()
ax6.set_title(f"{stock} close + model 2 regimes")

ax5.set_xlabel("time")
ax5.set_ylabel("return")
ax5.plot(test_df.index[split_data:], test_df['z log return 5d'][split_data:], label='returns')
ax5.axhline(0, color="black", linestyle="--")
ax5.set_title(f"{stock} daily returns")

ax7.set_xlabel("time")
ax7.set_ylabel("macd_rolling_z")
ax7.plot(test_df.index[split_data:], test_df['macd_rolling_z'][split_data:], label='macd_rolling_z')


# online learning
window = 100
value = 1
close_prices = np.array(test_df['Close'].iloc[split_data:])
b = 0
returns = list()
daily_returns = list()
Q = list()
state = list()
j = 0
j2 = 0

accuracy = 0
b_profit = 0

model = None

bull = None
bear = None

startprob = None
transmat = None
weights = None
means = None
covars = None

for i in range(window - 1, len(close_prices) - 1):

    O1 = df[["z log return 5d", "volatility_rolling", "price vs ma50", "macd_rolling_z", "beta", "vol_of_vol"]].values
    O_train = O1[j:split_data + j]

    scaler = PCA(whiten=True)
    O_train = scaler.fit_transform(O_train)

    if j % 5 == 0:
        print(f"train iter {j}")
        if j == 0:
            best_likelihood = -np.inf
            base_seed = random.randint(1, 10000)
            for k in range(100):
                print(f"random init {k}")
                curr_model = GMMHMM(n_components=2, n_mix=3, covariance_type="full", n_iter=15, random_state=base_seed + 2 * k)
                curr_likelihood = 0
                try:
                    curr_model.fit(O_train)
                    curr_likelihood = curr_model.score(O_train)
                except ValueError:
                    print("skipping init, nonpositive definite covariance matrix")
                else:
                    if curr_likelihood > best_likelihood:
                        best_likelihood = curr_likelihood
                        model = curr_model
            startprob = model.startprob_
            transmat = model.transmat_
            weights = model.weights_
            means = model.means_
            covars = model.covars_
        else:
            update_model = GMMHMM(n_components=2, n_mix=3, covariance_type="full"
                                  , n_iter=2, init_params="")
            update_model.startprob_ = startprob
            update_model.transmat_ = transmat
            update_model.weights_ = weights
            update_model.means_ = means
            update_model.covars_ = covars
            try:
                update_model.fit(O_train)
            except ValueError:
                print("skipping update, nonpositive definite covariance matrix")
            else:
                startprob = update_model.startprob_
                transmat = update_model.transmat_
                weights = update_model.weights_
                means = update_model.means_
                covars = update_model.covars_
                # update the current model
                model = update_model

        state_avg_return = []
        for s in range(model.n_components):
            weighted_means = np.average(
                model.means_[s][:, 0],   
                weights=model.weights_[s]
            )
            state_avg_return.append(weighted_means)
        bull = np.argmax(state_avg_return)
        bear = np.argmin(state_avg_return)
    

    O2 = test_df[["z log return 5d", "volatility_rolling", "price vs ma50", "macd_rolling_z", "beta", "vol_of_vol"]].values
    O_test = O2[split_data + j: split_data + j + window]

    O_test = scaler.transform(O_test)

    log_prob_window, Q_window = model.decode(O_test, algorithm="viterbi")
    post = model.predict_proba(O_test[-1:])[0]    # just the last observation in the window
    p_bull = post[bull]
    pos = 1 if p_bull > 0.6 else 0
    # if pos:
    if Q_window[-1] == bull:
        profit = 1 + ((close_prices[i + 1] - close_prices[i]) / close_prices[i])
        daily_returns.append(profit - 1)
        value *= profit
        b+=1
        b_profit += 1 if profit - 1 > 0 else 0
        Q.append(1)
    else:
        # sharpe ratio should include/not include this, depending on purpose
        # daily_returns.append(0)
        Q.append(0)

    returns.append((value - 1) * 100)

    j+=1

    states = model.predict(O_test)  
    if len(np.unique(states)) > 1:
        logreg = LogisticRegression().fit(O_test, states)  
        p = logreg.predict(O_test)
        accuracy += accuracy_score(states, p)
        j2+=1
    
print(f"results similar to a logistic regression by {accuracy / j2}")

q_t = Q[0]
start = 0
for t in range(1, len(Q)):
    if Q[t] != q_t or t == len(Q) - 1:
        end = t
        color = 'green' if q_t == 1 else 'red'
        ax6.axvspan(test_df.index[split_data + window + start], test_df.index[split_data + window + end], color=color, alpha=0.3)
        q_t = Q[t]
        start = t

temp = len(test_df) - len(Q)
Q = [np.nan] * temp + list(Q)
test_df['Q'] = Q
print(test_df.groupby('Q')[['z log return 5d', 'volatility_rolling', 'price vs ma50', 'macd_rolling_z', 'beta', 'vol_of_vol']].mean())

print()
print(f"{j} trading days")
print(f"cumulative return: {(value - 1):.2%}.")
print(f"annualized return: {(value ** (252/j) - 1):.2%}.")
print(f"bought {b / j:.2%} of the time")
print(f"profitable {b_profit / b:.2%} of the time")
daily_returns = np.array(daily_returns)
if np.std(daily_returns) != 0:
    sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
else:
    sharpe_ratio = np.nan
print(f"sharpe ratio: {sharpe_ratio:.2f}")

fig4, ax12 = plt.subplots(figsize=(10, 5))
ax12.plot(test_df.index[split_data + window:], returns, color='green')

###### Let's compare to the baseline
df4 = data2.xs(stock, level=1, axis=1).copy()
test_start_date = test_df.index[split_data + window]
test_stock_df = df4.loc[test_start_date:]
test_stock_cumulative = (test_stock_df['Close'].iloc[-1] - test_stock_df['Close'].iloc[0]) / test_stock_df['Close'].iloc[0]
test_stock_days = len(test_stock_df)
print()
print("Baseline return")
print(f"cumulative return: {(test_stock_cumulative):.2%}.")
print(f"annualized return: {((1 + test_stock_cumulative) ** (252/test_stock_days) - 1):.2%}.")

###### Let's test how our annualized returns compare to the SP500
print()
test_start_date = test_df.index[split_data + window]
spy_test_df = spy_df.loc[test_start_date:]
spy_cumulative = (spy_test_df['Close'].iloc[-1] - spy_test_df['Close'].iloc[0]) / spy_test_df['Close'].iloc[0]
spy_days = len(spy_test_df)
print("SPY return")
print(f"cumulative return: {(spy_cumulative):.2%}.")
print(f"annualized return: {((1 + spy_cumulative) ** (252/spy_days) - 1):.2%}.")
ax13.plot(spy_df.index[split_data + window:], spy_df['Close'].iloc[split_data + window:])
ax13.set_title("SPY close (benchmark)")

print(f"\n{base_seed}")

plt.show()
