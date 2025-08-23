import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = yf.download("SPY", period="2y")

# R_t = log(P_t / P_{t - 1}) = log(P_t) - log(P_{t - 1})
R_t_temp = np.log(np.array(df["Close"]))
df["Log Close"] = R_t_temp
R_t = np.diff(np.log(np.array(df["Close"])), axis=0)
df["Log Return"] = np.append([np.nan], R_t)
Vol_t = df["Log Return"].rolling(20).std()
df["Rolling Volatility"] = Vol_t
df = df.dropna()

# kalman filter logic
w_t = np.array(df.shape[0] * [0.0])
for i in range(len(w_t)):
    w_t[i] = np.random.normal(0, 1.0)

x_t = np.array(df.shape[0] * [[0.0]])
x_t[0] = np.array([[df["Log Close"].iloc[0]]])

P_t = np.array(df.shape[0] * [[0.0]])
P_t[0] = np.array([[1e-1]])

for i in range(1, len(x_t)):
    Q_t = np.array([[df["Rolling Volatility"].iloc[i] ** (1.3)]])
    alpha = 1e2
    R_t = np.array([[alpha * Q_t]])
    # R_t = np.array([[8e-3]])
    F_t = np.array([[1.0]])
    H_t = np.array([[1.0]])
    # prior
    x_t[i] = x_t[i - 1]
    P_t[i] = Q_t + F_t @ P_t[i - 1] @ F_t.T

    # kalman gain
    S_t = H_t @ P_t[i] @ H_t.T + R_t
    K_t = P_t[i] @ H_t.T @ np.linalg.inv(S_t)

    # posterior
    z_t = df["Log Close"].iloc[i]
    x_t[i] = x_t[i] + K_t @ (z_t - H_t @ x_t[i])
    P_t[i] = (np.eye(1) - K_t @ H_t) @ P_t[i] 



# plot figures
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.set_title("Log Close")
ax1.plot(df.index, df["Log Close"], 'b', df.index, x_t.ravel(), 'r')
ax1.fill_between(df.index, np.subtract(x_t.ravel(), P_t.ravel()), np.add(x_t.ravel(), P_t.ravel()), alpha=0.2)

ax3 = fig.add_subplot(212)
ax3.set_title("Rolling Volatility")
ax3.plot(df.index, df["Rolling Volatility"])

fig.tight_layout()

plt.show()