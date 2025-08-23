import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
import math
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import percentileofscore
import random
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed()

data_spy = yf.download('SPY', period='16y', auto_adjust=True)
df_spy = data_spy.xs('SPY', level=1, axis=1).copy()
df_spy['log return'] = np.log(df_spy['Close']).diff()

stock_train = 'AMD'
stock = 'AMD'

macd_window = 30
vol_window = 20

###### Initial stock is what we train the model with. 
data = yf.download(stock_train, period='16y', auto_adjust=True)
df = data.xs(stock_train, level=1, axis=1).copy()

# Preprocess the fetched stock data and use TA library to calculate indicators
stock_close = data['Close'][stock_train]
stock_returns = np.log(stock_close).diff()
stock_volatility = stock_returns.rolling(vol_window).std()

df = ta.add_all_ta_features(
    df,
    open="Open",
    high="High",
    low="Low",
    close="Close",
    volume="Volume",
    fillna=True
)

df['log return'] = stock_returns
df['volatility'] = stock_volatility
df['volatility_rolling'] = df['volatility'].rolling(window=vol_window).mean()
df['vol_of_vol'] = df['volatility_rolling'].rolling(window=vol_window).std()
df["macd"] = df["trend_macd_diff"]
df['macd_rolling_z'] = (df['macd'] - df['macd'].rolling(macd_window).mean()) / df['macd'].rolling(macd_window).std()
df['log return 5d'] = np.log(df['Close'] / df['Close'].shift(5))
rolling_mean = df['log return 5d'].rolling(window=60).mean()
rolling_std = df['log return 5d'].rolling(window=60).std()
df['z log return 5d'] = (df['log return 5d'] - rolling_mean) / rolling_std
df['price vs ma100'] = df['Close'] / df['Close'].rolling(60).mean() - 1
rolling_cov = df['log return'].rolling(60).cov(df_spy['log return'])
rolling_var = df_spy['log return'].rolling(60).var()
df['beta'] = rolling_cov / rolling_var
df = df.dropna()

######## Now we process the test stock
data2 = yf.download(stock, period='16y', auto_adjust=True)
df2 = data2.xs(stock, level=1, axis=1).copy()

# Preprocess the second stock (testing stock)
stock_close = data2['Close'][stock]
stock_returns = np.log(stock_close).diff()
stock_volatility = stock_returns.rolling(vol_window).std()

df2 = ta.add_all_ta_features(
    df2,
    open="Open",
    high="High",
    low="Low",
    close="Close",
    volume="Volume",
    fillna=True
)

df2['log return'] = stock_returns
df2['volatility'] = stock_volatility
df2['volatility_rolling'] = df2['volatility'].rolling(window=vol_window).mean()
df2['vol_of_vol'] = df2['volatility_rolling'].rolling(window=vol_window).std()
df2["macd"] = df2["trend_macd_diff"]
df2['macd_rolling_z'] = (df2['macd'] - df2['macd'].rolling(macd_window).mean()) / df2['macd'].rolling(macd_window).std()
df2['log return 5d'] = np.log(df2['Close'] / df2['Close'].shift(5))
rolling_mean = df2['log return 5d'].rolling(window=60).mean()
rolling_std = df2['log return 5d'].rolling(window=60).std()
df2['z log return 5d'] = (df2['log return 5d'] - rolling_mean) / rolling_std
df2['price vs ma100'] = df2['Close'] / df2['Close'].rolling(60).mean() - 1
rolling_cov = df2['log return'].rolling(60).cov(df_spy['log return'])
rolling_var = df_spy['log return'].rolling(60).var()
df2['beta'] = rolling_cov / rolling_var
df2 = df2.dropna()

# make sure that spy lines up with 
df_spy = df_spy[df_spy.index.isin(df2.index)]

#### Some simple functions, you can ignore

def log_return(P):
    return np.diff(np.log(P))

def compute_vol(P, window):
    vol = []
    for i in range(window, len(P) + 1):
        curr = P[i - window:i]
        vol.append(np.std(curr))
    return vol

def simple_return(P):
    return (P[1:] - P[:-1]) / P[:-1]

##### Preprocess and resegment data (training/testing)
split_data = 2750

######### Now we fit a gaussian emission HMM using hmmlearn library to the data 

###### Create the second model
# model2 = GMMHMM(n_components=2, covariance_type="full", n_iter=20, init_params="stmc")

# model2.startprob_ = np.array([0.0, 1.0])
# model2.transmat_ = np.array([[0.98103953, 0.01896047], [0.032724,   0.967276  ]])

# model2.means_ = np.array([[ 0.05477038, -0.58546092,  0.0820837,   0.03593272],   
#                          [-0.08993659,  0.96136546, -0.13478685, -0.0590039 ]])   

# bear_cov2 = np.array([[ 0.49384567,  0.04984693,  0.47534946,  0.27558166],
#   [ 0.04984693,  0.15002338,  0.02066328,  0.00670609],
#   [ 0.47534946,  0.02066328,  0.94502574,  0.37961371],
#   [ 0.27558166,  0.00670609,  0.37961371,  0.61126753]])

# bull_cov2 = np.array([[ 1.81815958,  0.14907953,  0.9606463,   0.81327155],
#   [ 0.14907953,  0.90868747, -0.02587144,  0.20363645],
#   [ 0.9606463,  -0.02587144,  1.0610753,   0.65014035],
#   [ 0.81327155,  0.20363645,  0.65014035,  1.63275796]])

# model2.covars_ = np.array([bear_cov2, bull_cov2])


###### Preprocess the observation sequence and train model 2
O_5d = df[["z log return 5d", "volatility_rolling", "price vs ma100", "macd_rolling_z", "beta", "vol_of_vol"]].values
O_5d_2 = df2[["z log return 5d", "volatility_rolling", "price vs ma100", "macd_rolling_z", "beta", "vol_of_vol"]].values
O_5d_train = O_5d[:split_data]
O_5d_test = O_5d_2[split_data:]
# scaler = StandardScaler()
scaler = PCA(whiten=True)
O_5d_train_scaled = scaler.fit_transform(O_5d_train)
O_5d_test_scaled = scaler.transform(O_5d_test)
model2 = None
best_log_likelihood = -np.inf
base_seed = 20491
# base_seed = random.randint(0, 100000)
for i in range(5):
    model = GMMHMM(n_components=2, n_mix=8, covariance_type="full", n_iter=15, init_params="stmcw", random_state=base_seed + i)
    model.fit(O_5d_train_scaled)
    score = model.score(O_5d_train_scaled)
    if score > best_log_likelihood:
        best_log_likelihood = score
        model2 = model
model2.fit(O_5d_train_scaled)

# Assume feature 0 is z log return 5d
state_avg_return = []

for s in range(model.n_components):
    weighted_means = np.average(
        model.means_[s][:, 0],    # get the means of the 0th feature (z log return 5d)
        weights=model.weights_[s]
    )
    state_avg_return.append(weighted_means)

# Determine which state corresponds to higher expected return
bull_state = np.argmax(state_avg_return)
bear_state = np.argmin(state_avg_return)

print("Bull state:", bull_state)
print("Bear state:", bear_state)

#### Makes second subplot
fig2, (ax5, ax6, ax7, ax8, ax9, ax13, ax14, ax15) = plt.subplots(
    8, 1,
    figsize=(11, 10),
    sharex=True,
    gridspec_kw={'hspace': 0.7} 
)

fig2.suptitle(f"bull/bear market regimes for {stock}\nusing gaussian density hmm (model 2)", fontsize=14, fontweight='bold')

ax8.set_xlabel("time")
ax8.plot(df2.index[split_data:], df2['price vs ma100'][split_data:], label='price vs ma100')
ax8.set_title("price vs ma100")

ax14.set_xlabel("time")
ax14.plot(df2.index[split_data:], df2['beta'][split_data:], label='beta')
ax14.set_title("beta")

ax9.plot(df2.index[split_data:], df2['volatility_rolling'][split_data:], label='volatility_rolling')
vol_mean = df2['volatility_rolling'].mean()
ax9.axhline(vol_mean)
ax9.set_xlabel("time")
ax9.set_ylabel("volatility_rolling")
ax9.set_title(f"{stock} vol")
ax9.legend()

ax15.plot(df2.index[split_data:], df2['vol_of_vol'][split_data:], label='vol_of_vol')
ax15.set_xlabel("time")
ax15.set_ylabel("vol_of_vol")
ax15.set_title(f"{stock} vol of vol")
ax15.legend()

ax6.set_xlabel("time")
ax6.set_ylabel("price")
ax6.plot(df2.index[split_data:], df2['Close'][split_data:], label='close')
ax6.legend()
ax6.set_title(f"{stock} close + model 2 regimes")
ax6.text(0.01, 0.95, f"log likelihood {model2.score(O_5d_test_scaled)}", 
         transform=ax6.transAxes,
         verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax5.set_xlabel("time")
ax5.set_ylabel("return")
ax5.plot(df2.index[split_data:], df2['z log return 5d'][split_data:], label='returns')
ax5.axhline(0, color="black", linestyle="--")
ax5.set_title(f"{stock} daily returns")

ax7.set_xlabel("time")
ax7.set_ylabel("macd_rolling_z")
ax7.plot(df2.index[split_data:], df2['macd_rolling_z'][split_data:], label='macd_rolling_z')

#### Print model 2's parameters and useful data post-fit to the console
print("\nModel 2 Parameters:")
print("- initial state distribution")
print(np.array(model2.startprob_))
print("- transition distribution")
print(np.array(model2.transmat_))
print("- means of emissions")
print(np.array(model2.means_))
print("- covars of emissions")
print(np.array(model2.covars_))
print()

### model 2 posteriors
fig3, (ax11) = plt.subplots(1, 1, figsize=(8, 2), sharex=True, gridspec_kw={'hspace': 0.4} )
posterior_probs = model2.predict_proba(O_5d_test_scaled)
state0_probs = posterior_probs[:, 1 - bull_state]
state1_probs = posterior_probs[:, bull_state]
ax11.stackplot(df2.index[split_data:], [state0_probs, state1_probs],
             labels=['bear', 'bull'],
             colors=['red', 'green'],
             alpha=0.3)
ax11.set_title(f"posteriors for model 2")
ax11.set_ylabel("prob")
ax11.legend(loc='upper left')
ax11.set_ylim(0, 1)

# I will now calculate the profit using a sliding window method
# I initialize a 50d trading day period (slightly over a month)
# and iteratively calculate the most likely state sequence with
# the viterbi procedure. Then, at each produced sequence I 
# simulate the naiive trading strategy of buying the close price
# and selling the close price on the next day if a bull state is 
# is predicted, or doing nothing if a bear state is predicted. 
# finally, I sum the profit made in this manner over 
# the (period of time) - (window of time), i.e. profit is
# calculated over all data points where a past window 
# of time exists. 
window = 60
value = 1
baseline_value = 1
close_prices = np.array(df2['Close'].iloc[split_data:])
bought = 0
days = 0
returns = list()
daily_returns = list()
viterbi_Q_5d = list()
for i in range(window - 1, len(O_5d_test_scaled) - 1):
    O_window = O_5d_test_scaled[i - window + 1:i + 1]
    log_prob_window, Q_window = model2.decode(O_window, algorithm="viterbi")
    if Q_window[-1] == bull_state:
        profit = 1 + ((close_prices[i + 1] - close_prices[i]) / close_prices[i])
        daily_returns.append(profit - 1)
        value *= profit
        baseline_value *= profit
        bought+=1
        viterbi_Q_5d.append(bull_state)
    else:
        # sharpe ratio should include/not include this, depending on purpose
        # daily_returns.append(0)
        viterbi_Q_5d.append(bear_state)
        profit = 1 + ((close_prices[i + 1] - close_prices[i]) / close_prices[i])
        baseline_value *= profit
    returns.append((value - 1) * 100)
    days+=1

current_state = viterbi_Q_5d[0]
start = 0
for t in range(1, len(viterbi_Q_5d)):
    if viterbi_Q_5d[t] != current_state or t == len(viterbi_Q_5d) - 1:
        end = t 
        color = 'green' if current_state == bull_state else 'red'
        ax6.axvspan(df2.index[split_data + window + start], df2.index[split_data + window + end], color=color, alpha=0.3)
        current_state = viterbi_Q_5d[t]
        start = t

padding_length = len(df2) - len(viterbi_Q_5d)
viterbi_Q_full = [np.nan] * padding_length + list(viterbi_Q_5d)
df2['viterbi model 2'] = viterbi_Q_full
print(df2.groupby('viterbi model 2')[['z log return 5d', 'volatility_rolling', 'price vs ma100', 'macd_rolling_z', 'beta', 'vol_of_vol']].mean())

print()
print(f"****stats over {days} trading days****")
print(f"trained with {df2.index[0].date()} to {df2.index[split_data - 1].date()}.")
print(f"backtested with {df2.index[split_data + window].date()} to {df2.index[-1].date()}.")
print(f"cumulative return of {(value - 1):.2%}.")
print(f"annualized return of {(value ** (252/days) - 1):.2%}.")
print(f"there was a {bought / days:.2%} chance you bought on a particular day.")
daily_returns = np.array(daily_returns)
if np.std(daily_returns) != 0:
    sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
else:
    sharpe_ratio = np.nan
print(f"sharpe ratio of: {sharpe_ratio:.2f}")

fig4, ax12 = plt.subplots(figsize=(10, 5))
fig4.suptitle(f"percent returns from {df2.index[split_data + window].date()} to {df2.index[-1].date()}")
ax12.plot(df2.index[split_data + window:], returns, color='green')

###### Let's compare to the baseline
test_cumulative = (df2['Close'].iloc[-1] - df2['Close'].iloc[split_data + window]) / df2['Close'].iloc[split_data + window]
test_days = len(df2) - (split_data + window)
print()
print("Baseline returns:")
print(f"return captured from {df2.index[split_data + window].date()} to {df2.index[-1].date()}.")
print(f"cumulative return of {(test_cumulative):.2%}.")
print(f"annualized return of {((1 + test_cumulative) ** (252/test_days) - 1):.2%}.")
print(f"if you had bought/sold everyday: {(baseline_value - 1):.2%}.")

###### Let's test how our annualized returns compare to the SP500
print()
spy_cumulative = (df_spy['Close'].iloc[-1] - df_spy['Close'].iloc[split_data + window]) / df_spy['Close'].iloc[split_data + window]
spy_days = len(df_spy) - (split_data + window)
print("SPY returns:")
print(f"return captured from {df_spy.index[split_data + window].date()} to {df2.index[-1].date()}.")
print(f"cumulative return of {(spy_cumulative):.2%}.")
print(f"annualized return of {((1 + spy_cumulative) ** (252/spy_days) - 1):.2%}.")
ax13.set_title("SPY close (benchmark)")
ax13.plot(df_spy.index[split_data + window:], df_spy['Close'].iloc[split_data + window:])

print(f"\nbase seed for replication: {base_seed}")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

states = model2.predict(O_5d_test_scaled)  # HMM-predicted hidden states (0 or 1)

clf = LogisticRegression().fit(O_5d_test_scaled, states)  # train classifier to mimic HMM
pred_states = clf.predict(O_5d_test_scaled)

acc = accuracy_score(states, pred_states)
print("logistic regression acc vs hmm:", acc)

if value - 1 > baseline_value - 1 and value - 1 > test_cumulative and value - 1 > spy_cumulative:
    print("ALPHALPAALPHALPAALPHALPA")
    print(f"you beat the naiive strategy by {value - baseline_value:.2%}")

plt.show()