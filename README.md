# AR-GAS-Regime-Model

<img width="1086" height="666" alt="image" src="https://github.com/user-attachments/assets/2cfe82b9-3ea9-49dd-a65a-cf46dbf77406" />

The resulting state segmentation of 40-iter EM of HamiltonGAS on SPY. We train on log returns from 2010-01-01 to 2019-01-01 and the test (shown) on 2019-03-01 to 2025-01-01. The colors represent the segmented states learned by the model.

Parameters: 

  State        μ       φ₁       φ₂        ω       A       B
-------  -------  -------  -------  -------  ------  ------
      0   0.0014  -0.0761  -0.0208  -0.6522  0.0748  0.9390
      1  -0.0219   0.0733  -0.2983  -0.5046  0.0006  0.9900
      2  -0.0012  -0.0442  -0.0039  -1.6499  0.2492  0.8074


- Creal, D., Koopman, S. J., & Lucas, A. (2013). Generalized Autoregressive Score Models with Applications. Journal of Applied Econometrics, 28(5), 777–795. https://doi.org/10.1002/jae.1279

- Hamilton, J. D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. Econometrica, 57(2), 357–384. https://doi.org/10.2307/1912559

- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition. Proceedings of the IEEE, 77(2), 257–286. https://doi.org/10.1109/5.18626
