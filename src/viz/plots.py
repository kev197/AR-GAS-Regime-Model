import yfinance as yf
import math
import numpy as np
import pandas as pd
import ta
import matplotlib.pyplot as plt 
from tabulate import tabulate
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from src.data.preprocessing import train_df, test_df, spy_df
from src.models.hmm_ar import Hamilton
from src.models.hmm_ar_gas import HamiltonGAS

y = np.array(train_df['Log return'])
model = HamiltonGAS(NSTATES=3, r = 2, p = 0, q = 0, PHI=[0.2, 0.7])
model.print_modelparams()
T = 40
loglikelis = list()
for i in range(T):
    model.EM(y)
    print(f"EM{i}")
    model.print_modelparams()
    if i >= 3:
        loglikelis.append(model.print_loglikeli())

y = np.array(test_df['Log return'])

def show():
    fig = plt.figure()
    fig.set_size_inches(11, 8)

    def showsim():
        f = plt.figure()
        sim = model.simulatemodels()
        m1 = f.add_subplot(311)
        m1.set_title("State 0 AR")
        m1.plot(sim[0])
        m2 = f.add_subplot(312)
        m2.set_title("State 1 AR")
        m2.plot(sim[1])
        m3 = f.add_subplot(313)
        m3.set_title("State 2 AR")
        m3.plot(sim[2])

    # ax1 = fig.add_subplot(211)
    # ax1.set_title("Log return")
    # ax1.plot(train_df.index, train_df["Log return"])

    ax1 = fig.add_subplot(212)
    ax1.set_title(f"Log Likelihood from iter 3 to {T}")
    ax1.plot(loglikelis)

    ax2 = fig.add_subplot(211)
    ax2.set_title("Prediction Posteriors Hamilton AR(2)")
    model.FBP(y)
    posteriors = np.array(model.getposteriors())

    states = list()
    for t in range(2, len(y) - 1):
        model.FBP(y[:t + 1])
        pred = model.forecast()
        states.append(pred)

    states = np.concatenate([np.full(3, -1), states])

    # states = posteriors.argmax(axis=0)
    # states = np.concatenate([np.full(2, -1), states])

    cols = ["#2d86c6", "#e02121", "#28b028"]
    for s in range(model.getnstates()):
        idx = states == s
        ax2.scatter(test_df.index[idx], 
                test_df["Close"].iloc[idx], 
                color=cols[s], s=11, label=f"state {s}", alpha=0.7)
    ax2.grid(True, alpha=0.3, linestyle="--")

    ax2.plot(test_df.index, test_df["Close"], 'gray', linewidth=0.2, alpha=0.8)

    # ax3 = fig.add_subplot(313)
    # ax3.set_title("Volatility")
    # ax3.plot(train_df.index, train_df["Volatility"])

    fig.tight_layout()

    plt.show()

show()