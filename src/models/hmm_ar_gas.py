import yfinance as yf
import math
import numpy as np
import pandas as pd
import ta
import matplotlib.pyplot as plt 
import copy
from tabulate import tabulate
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from scipy.optimize import minimize

from src.data.preprocessing import train_df, test_df, spy_df

class HamiltonGAS:
    def __init__(self, NSTATES, r, p, q, PHI):
        self.NSTATES = NSTATES
        self.r = r
        self.p = p
        self.q = q
        self.PHI = np.array([[0.5, 0.7], 
                           [0.4, 0.25], 
                           [0.2, 0.9]])
        self.mu = [0.03, -0.03, 0]
        self.P = np.array([[0.9, 0.05, 0.05], 
                           [0.05, 0.9, 0.05], 
                           [0.05, 0.05, 0.9]])
        self.pi = [0.3, 0.3, 0.4]
        self.alpha = self.beta = self.gamma = self.xi = self.c = list()

        self.STDEST = [0.2, 0.2, 0.2]
        self.A = [0.1, 0.1, 0.1]
        self.B = [0.5, 0.5, 0.5]
        self.omega = list()
        for i in range(self.NSTATES):
            self.omega.append((1 - self.B[i]) * np.log(self.STDEST[i]))
        self.f = list()
        self.s = list()
        self.initGAS()

    def initGAS(self):
        SENTINEL = -1
        self.f = list()
        for i in range(self.NSTATES):
            self.f.append(list())
            self.f[i].append(SENTINEL)
            self.f[i].append(np.log(self.STDEST[i]))
        self.s = list()
        for i in range(self.NSTATES):
            self.s.append(list())
            self.s[i].append(SENTINEL)
            self.s[i].append(0)

    def forecast(self):
        if len(self.alpha[0]) == 0 or len(self.alpha[0]) != len(self.beta[0]):
            print("forecast failed, bad fbp")
            return
        forecasted = list()
        for i in range(self.NSTATES):
            summation = 0
            for j in range(self.NSTATES):
                summation += self.alpha[j][-1] * self.P[j][i]
            forecasted.append(summation)
        forecasted = np.array(forecasted)
        forecasted /= forecasted.sum()
        return forecasted.argmax(axis=0)
    
    def computeAR(self, state, lagged1, lagged2):
        return self.mu[state] + lagged1 * self.PHI[state][0] + lagged2 * self.PHI[state][1]
    
    def computeGAS(self, state, t):
        return self.omega[state] + self.A[state] * self.s[state][t - 1] + self.B[state] * self.f[state][t - 1]
    
    def propagate_gas(self, y, state, t):
            gas = self.computeGAS(state, t)
            self.f[state].append(gas)
            return gas

    def propagate_score(self, y, state, t):
        resid = y[t] - self.computeAR(state, y[t - 1], y[t - 2])
        gradient = -(1/2) + (1/2) * np.exp(-self.f[state][t]) * (resid**2)
        score = gradient
        self.s[state].append(score)
        return score
    
    def getposteriors(self):
        return self.gamma
    
    def getnstates(self):
        return self.NSTATES

    def gettransmat(self):
        return self.P
    
    def print_transmat(self):
        for i in range(self.NSTATES):
            print(f"{'%f' % self.P[i][0]} {'%f' % self.P[i][1]} {'%f' % self.P[i][2]}")
    
    def simulatemodels(self):
        simulations = list()
        for i in range(self.NSTATES):
            simulation = [0,0]
            # TODO: fix this
            std = 1
            for t in range(200):
                simulation.append(self.computeAR(i, simulation[-1], simulation[-2]) + np.random.normal(0, std))
            simulations.append(simulation)
        return simulations
    
    def print_modelparams(self):
        t = []
        for i in range(self.NSTATES):
            mu = float(self.mu[i])
            phi1, phi2 = self.PHI[i]
            omega = self.omega[i]
            A = self.A[i]
            B = self.B[i]
            t.append([f"{i}", mu, phi1, phi2, omega, A, B])
        print(tabulate(t, headers=["State", "μ", "φ₁", "φ₂", "ω", "A", "B"], floatfmt=".4f"))

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
                self.P[i][j] = numer / denom
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

            # TODO fix this with GAS MLE
            assert (len(y) - 2) == len(self.gamma[i])
    #         SCIPY_BFGS = minimize(self.gas_surrogate_loglikeli, x0=[self.A[i], self.B[i], self.omega[i]], args=(i, y)
    #                               , method='L-BFGS-B', bounds=[(0, 5), (0.1, 0.95), (-1, 1)],
    # options={'maxiter': 100, 'ftol': 1e-8})
            transformparams = [np.log(self.A[i]), np.log(self.B[i] / (0.99 - self.B[i])), self.omega[i]]        
            SCIPY_BFGS = minimize(self.gas_surrogate_loglikeli, x0=transformparams, args=(i, y)
                                  , method='BFGS')
            correctparams = self.transform_params(SCIPY_BFGS.x)
            self.A[i], self.B[i], self.omega[i] = correctparams

    def computeGAS_minimize(self, state, t, gasparams):
        A, B, omega = gasparams
        return omega + A * self.s[state][t - 1] + B * self.f[state][t - 1]

    def propagate_gas_minimize(self, y, state, t, gasparams):
            gas = self.computeGAS_minimize(state, t, gasparams)
            self.f[state].append(gas)
            return gas

    def propagate_score_minimize(self, y, state, t):
        resid = y[t] - self.computeAR(state, y[t - 1], y[t - 2])
        gradient = -(1/2) + (1/2) * np.exp(-self.f[state][t]) * (resid**2)
        score = gradient
        self.s[state].append(score)
        return score
    
    def transform_params(self, gasparams):
        A = np.exp(gasparams[0])  
        B = 0.99 * (1 / (1 + np.exp(-gasparams[1])))  
        omega = gasparams[2]  
        return A, B, omega

    def gas_surrogate_loglikeli(self, gasparams, state, y):
        # transform unconstrained theta into valid params
        A, B, omega = self.transform_params(gasparams)
        gasparams = (A, B, omega)

        self.initGAS()
        
        for t in range(2, len(y)):
            self.propagate_gas_minimize(y, state, t, gasparams)
            self.propagate_score_minimize(y, state, t)

        assert len(self.f[state]) == len(y)

        A, B, omega = gasparams

        Q = 0
        for t in range(2, len(y)):
            resid = y[t] - self.computeAR(state, y[t - 1], y[t - 2])
            loglikeli = -(1/2) * (np.log(2 * np.pi) + self.f[state][t] + np.exp(-self.f[state][t]) * (resid**2))
            Q += self.gamma[state][t - 2] * loglikeli

        return -Q

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
        
        f = [fstate[2:] for fstate in copy.deepcopy(self.f)]

        for t in range(n - 1):
            summation = 0
            b = [[0.0 for _ in range(self.NSTATES)] for _ in range(self.NSTATES)]
            for i in range(self.NSTATES):
                for j in range(self.NSTATES):
                    gas = f[j][t + 1]
                    std = np.exp(0.5 * gas)
                    predicted = self.computeAR(j, y[t], y[t + 1])
                    b[i][j] = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((y[t + 1] - predicted)**2) / (2 * (std**2)))
                    summation += self.alpha[i][t] * self.P[i][j] * b[i][j] * self.beta[j][t + 1]
            for i in range(self.NSTATES):
                for j in range(self.NSTATES):
                    joint[i][j].append((self.alpha[i][t] * self.P[i][j] * b[i][j] * self.beta[j][t + 1]) / summation)
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
                    gas = self.f[j][t + 1]
                    std = np.exp(0.5 * gas)
                    predicted = self.computeAR(j, y[t], y[t - 1])
                    b = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((y[t + 1] - predicted)**2) / (2 * (std**2)))
                    summation += self.P[i][j] * b * backward_vars[j][t + 1]
                induction = summation
                backward_vars[i][t] = induction
            rescale(t)

        backward_vars = [bwd_var[2:] for bwd_var in backward_vars]

        return backward_vars

    def forward(self, y):
        # implement forward-backward procedure from scratch (rabiner)
        forward_vars = list()
        c = [0.0, 1.0]

        self.initGAS()

        for i in range(self.NSTATES):
            forward_vars.append(list())
            gas = self.propagate_gas(y, i, 2)
            std = np.exp(0.5 * gas)

            predicted = self.computeAR(i, y[1], y[0])
            b = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((y[2] - predicted)**2) / (2 * (std**2)))
            basecase = self.pi[i] * b
            sentinel = np.inf
            forward_vars[i].append(sentinel)
            forward_vars[i].append(sentinel)
            forward_vars[i].append(basecase)

            score = self.propagate_score(y, i, 2)
        
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
                    summation += forward_vars[j][t - 1] * self.P[j][i]
                gas = self.propagate_gas(y, i, t)
                std = np.exp(0.5 * gas)

                predicted = self.computeAR(i, y[t - 1], y[t - 2])
                b = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((y[t] - predicted)**2) / (2 * (std**2)))
                induction = summation * b
                forward_vars[i].append(induction)

                score = self.propagate_score(y, i, t)
            rescale(t)

        forward_vars = [fwd_var[2:] for fwd_var in forward_vars]
        n = len(forward_vars[0])

        self.c = c

        return forward_vars, c
