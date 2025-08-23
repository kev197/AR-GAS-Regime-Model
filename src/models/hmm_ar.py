import yfinance as yf
import math
import numpy as np
import pandas as pd
import ta
import matplotlib.pyplot as plt 
from tabulate import tabulate
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from src.data.preprocessing import train_df, test_df, spy_df

class Hamilton:
    def __init__(self, NSTATES, r, p, q, PHI):
        self.NSTATES = NSTATES
        self.r = r
        self.p = p
        self.q = q
        self.PHI = np.array([[0.5, 0.4], 
                           [0.4, 0.2], 
                           [0.8, 0.9]])
        self.mu = [0.03, -0.03, 0]
        self.std = [0.02, 0.04, 0.03]
        self.A = np.array([[0.9, 0.05, 0.05], 
                           [0.05, 0.9, 0.05], 
                           [0.05, 0.05, 0.9]])
        self.pi = [0.3, 0.3, 0.4]
        self.alpha = self.beta = self.gamma = self.xi = self.c = list()

    def forecast(self):
        if len(self.alpha[0]) == 0 or len(self.alpha[0]) != len(self.beta[0]):
            print("forecast failed, bad fbp")
            return
        forecasted = list()
        for i in range(self.NSTATES):
            summation = 0
            for j in range(self.NSTATES):
                summation += self.alpha[j][-1] * self.A[j][i]
            forecasted.append(summation)
        forecasted = np.array(forecasted)
        forecasted /= forecasted.sum()
        return forecasted.argmax(axis=0)
    
    def compute(self, state, lagged1, lagged2):
        return self.mu[state] + lagged1 * self.PHI[state][0] + lagged2 * self.PHI[state][1]
    
    def getposteriors(self):
        return self.gamma
    
    def getnstates(self):
        return self.NSTATES

    def gettransmat(self):
        return self.A
    
    def print_transmat(self):
        for i in range(self.NSTATES):
            print(f"{'%f' % self.A[i][0]} {'%f' % self.A[i][1]} {'%f' % self.A[i][2]}")
    
    def simulatemodels(self):
        simulations = list()
        for i in range(self.NSTATES):
            simulation = [0,0]
            std = self.std[i]
            for t in range(200):
                simulation.append(self.compute(i, simulation[-1], simulation[-2]) + np.random.normal(0, std))
            simulations.append(simulation)
        return simulations
    
    def print_modelparams(self):
        t = []
        for i in range(self.NSTATES):
            mu = float(self.mu[i])
            phi1, phi2 = self.PHI[i]
            std = self.std[i]
            t.append([f"{i}", mu, phi1, phi2, std])
        print(tabulate(t, headers=["State", "μ", "φ₁", "φ₂", "σ"], floatfmt=".4f"))

    def print_loglikeli(self):
        log_likeli = 0
        for t in range(2, len(self.c)):
            log_likeli += np.log(self.c[t])
        log_likeli = -1 * log_likeli
        print(log_likeli)
        return log_likeli

    def EM(self, y):
        self.FBP(y)
        self.baumwelch(y)
    
    def baumwelch(self, y):
        if len(self.alpha[0]) == 0:
            print("FBP not init, EM failed")
            return
        # REMINDER: alpha is truncated by 2. 
        n = len(self.alpha[0])
        # optimize pi
        for i in range(self.NSTATES):
            self.pi[i] = self.gamma[i][0]
        # optimize transition matrix
        for i in range(self.NSTATES):
            for j in range(self.NSTATES):
                numer = denom = 0
                for t in range(n - 1):
                    numer += self.xi[i][j][t]
                    denom += self.gamma[i][t]
                self.A[i][j] = numer / denom
        # optimize model params
        Z = np.zeros((n, 3))
        for t in range(2, len(y)):
            Z[t - 2][0] = 1
            Z[t - 2][1] = y[t - 1]
            Z[t - 2][2] = y[t - 2]
        Y = y[2:]
        for i in range(self.NSTATES):
            W = np.diag(self.gamma[i])

            # A(beta) = b => np.linalg.solve(A, b)
            OPTIMIZED = np.linalg.solve(Z.T @ W @ Z, Z.T @ W @ Y)
            self.mu[i] = OPTIMIZED[0]
            self.PHI[i] = OPTIMIZED[1:]
            resid = Y - Z @ OPTIMIZED
            numer = 0
            denom = 0
            for t in range(n):
                numer += self.gamma[i][t] * (resid[t]**2)
                denom += self.gamma[i][t]
            OPTIMIZED = np.append(OPTIMIZED, np.sqrt(numer / denom))
            self.std[i] = OPTIMIZED[-1]


    def FBP(self, y):
        self.alpha, c = self.forward(y)
        self.beta = self.backward(y, c)
        self.gamma = self.posteriors()
        self.xi = self.joint_posteriors(y)

    def posteriors(self):
        if len(self.alpha[0]) == 0 or len(self.alpha[0]) != len(self.beta[0]):
            print("posterior generation failed, bad fbp")
            return
        n = len(self.alpha[0])
        posteriors = list()
        for i in range(self.NSTATES):
            posteriors.append(list())
        for t in range(n):
            summation = sum((self.alpha[i][t] * self.beta[i][t]) for i in range(self.NSTATES))
            for i in range(self.NSTATES):
                posteriors[i].append((self.alpha[i][t] * self.beta[i][t]) / summation)
        return posteriors
    
    def joint_posteriors(self, y):
        if len(self.alpha[0]) == 0 or len(self.alpha[0]) != len(self.beta[0]):
            print("joint posterior generation failed, bad fbp")
            return
        n = len(self.alpha[0])
        joint = list()
        for i in range(self.NSTATES):
            joint.append(list())
            for j in range(self.NSTATES):
                joint[i].append(list())
        for t in range(n - 1):
            summation = 0
            b = [[0.0 for _ in range(self.NSTATES)] for _ in range(self.NSTATES)]
            for i in range(self.NSTATES):
                for j in range(self.NSTATES):
                    std = self.std[j]
                    predicted = self.compute(j, y[t], y[t + 1])
                    b[i][j] = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((y[t + 1] - predicted)**2) / (2 * (std**2)))
                    summation += self.alpha[i][t] * self.A[i][j] * b[i][j] * self.beta[j][t + 1]
            for i in range(self.NSTATES):
                for j in range(self.NSTATES):
                    joint[i][j].append((self.alpha[i][t] * self.A[i][j] * b[i][j] * self.beta[j][t + 1]) / summation)
        return joint

    def backward(self, y, c):
        n = len(y)
        backward_vars = list()

        for i in range(self.NSTATES):
            backward_vars.append(list())
            backward_vars[i] = [0.0 for _ in range(n)]
            backward_vars[i][n - 1] = 1

        def rescale(t):
            nonlocal c
            for i in range(self.NSTATES):
                backward_vars[i][t] = c[t] * backward_vars[i][t] 
        rescale(n - 1)
    
        for t in range(n - 2, 1, -1):
            for i in range(self.NSTATES):
                summation = 0
                for j in range(self.NSTATES):
                    std = self.std[j]
                    predicted = self.compute(j, y[t], y[t - 1])
                    b = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((y[t + 1] - predicted)**2) / (2 * (std**2)))
                    summation += self.A[i][j] * b * backward_vars[j][t + 1]
                induction = summation
                backward_vars[i][t] = induction
            rescale(t)

        backward_vars = [bwd_var[2:] for bwd_var in backward_vars]

        return backward_vars

    def forward(self, y):
        # implement forward-backward procedure from scratch (rabiner)
        forward_vars = list()
        c = [0.0, 1.0]
        for i in range(self.NSTATES):
            forward_vars.append(list())
            std = self.std[i]
            predicted = self.compute(i, y[1], y[0])
            b = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((y[2] - predicted)**2) / (2 * (std**2)))
            basecase = self.pi[i] * b
            sentinel = np.inf
            forward_vars[i].append(sentinel)
            forward_vars[i].append(sentinel)
            forward_vars[i].append(basecase)
        
        def rescale(t):
            sum = 0
            for i in range(self.NSTATES):
                sum += forward_vars[i][t]
            nonlocal c
            c.append(1 / sum)
            for i in range(self.NSTATES):
                forward_vars[i][t] *= c[t]
        rescale(2)

        for t in range(3, len(y)):
            for i in range(self.NSTATES):
                summation = 0
                for j in range(self.NSTATES):
                    summation += forward_vars[j][t - 1] * self.A[j][i]
                std = self.std[i]
                predicted = self.compute(i, y[t - 1], y[t - 2])
                b = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((y[t] - predicted)**2) / (2 * (std**2)))
                induction = summation * b
                forward_vars[i].append(induction)
            rescale(t)

        forward_vars = [fwd_var[2:] for fwd_var in forward_vars]
        n = len(forward_vars[0])

        self.c = c

        return forward_vars, c
