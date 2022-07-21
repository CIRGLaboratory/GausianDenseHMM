from hmmlearn import hmm
import numpy as np
import os
# print(os.getcwd())
# os.chdir("/")
from models import GammaGaussianHMM

dense_hmm = GammaGaussianHMM(3)
## setup
n = 4
T = 100
np.random.seed(2022)

startprob = np.random.uniform(size=n)
startprob /= startprob.sum()

transmat = np.random.uniform(size=(n,  n))
transmat /= transmat.sum(axis=1)[:, np.newaxis]
means = np.random.randint(-10,  10, size=(n, 1))
covars = np.random.uniform(1, 10, (n, 1, 1))

## init model
model = hmm.GaussianHMM(n_components=n, covariance_type="full")

model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_ = covars

# sample
Y, X = model.sample(T)

## init params

startprob_init = np.random.uniform(size=n)
startprob_init /= startprob_init.sum()

transmat_init = np.random.uniform(size=(n,  n))
transmat_init /= transmat_init.sum(axis=1)[:, np.newaxis]
means_init = np.random.randint(-10,  10, size=(n, 1))
covars_init = np.random.uniform(1, 10, (n, 1, 1))


