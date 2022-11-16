import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf

import numpy as np
from tqdm import tqdm
from collections import deque

import time
import itertools
import wandb

import sklearn.cluster as cluster
from scipy.stats import multivariate_normal
from sklearn.tree import DecisionTreeClassifier

from hmmlearn import _utils
from hmmlearn.hmm import GaussianHMM, _check_and_set_gaussian_n_features
from hmmlearn.base import ConvergenceMonitor, check_array

from utils import pad_to_seqlen, check_random_state, dict_get, check_dir, check_arr, \
    empirical_coocs, iter_from_Xlengths, check_is_fitted, check_arr_gaussian, dtv, find_permutation, check_nodes

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from icecream import ic

class HMMLoggingMonitor(ConvergenceMonitor):

    def __init__(self, tol, n_iter, verbose, log_config=None, wandb_log=False, wandb_params=None, true_vals=None,
                 wandb_init=True, run=None):

        super(HMMLoggingMonitor, self).__init__(tol, n_iter, verbose)
        self.wandb_log = wandb_log
        self.accuracy = deque()
        self.omega_dtv = deque()
        self.z = deque()
        self.z0 = deque()
        self.u = deque()
        self.loss = deque()
        self._init_time = time.perf_counter()

        self.true_states = dict_get(true_vals, 'states')  # TODO: provide real values for evaluation
        self.true_transmat = dict_get(true_vals, 'transmat')  # TODO: use dict_get(dict, key, default=None)
        self.true_startprob = dict_get(true_vals, 'startprob')
        self.true_means = dict_get(true_vals, 'means')
        self.true_covars = dict_get(true_vals, 'covars')

        # Default log_config
        self.log_config = {
            'exp_folder': None,  # root experiment folder
            'log_folder': None,  # folder to store array-data in
            'metrics_initial': True,  # whether to compute metrics before first estep
            'metrics_after_mstep_every_n_iter': 1,  # frequency of computing metrics after mstep or none
            'metrics_after_estep_every_n_iter': 1,  # frequency of computing metrics after estep or none
            'metrics_after_convergence': False,  # whether to compute metrics after the training
            'gamma_after_estep': False,  # whether to compute gammas (they are very large!)
            'gamma_after_mstep': False,
            'test_gamma_after_estep': False,  # whether to compute test gammas ..
            'test_gamma_after_mstep': False,
            'samples_after_estep': None,  # (n_seqs, seqlen) sample to draw after estep or None
            'samples_after_mstep': None,  # (n_seqs, seqlen) sample to draw after mstep or None
            'samples_after_cooc_opt': None  # (n_seqs, seqlen) sample to draw after fitting model's coocs
        }

        self.run = None
        if log_config is not None:
            self.log_config.update(dict(log_config))

        # Default wandb parameters
        t = time.localtime()
        self.wandb_params = {
            "init": {
                "project": "gaussian-dense-hmm",
                "entity": "kabalce",
                "save_code": True,
                "group": f"benchmark-{t.tm_year}-{t.tm_mon}-{t.tm_mday}",
                "job_type": "n=0-s=0-T=0-simple=True",
                "name": "dense--l=0-lr=0-epochs=0"
            },
            "config": {
                "n": 0,
                "s": 0,
                "T": 0
            }
        }

        if wandb_params is not None:
            if "init" in wandb_params.keys():
                self.wandb_params["init"].update(wandb_params["init"])
            if "config" in wandb_params.keys():
                self.wandb_params["config"].update(wandb_params["config"])

        if wandb_init & wandb_log:
            self.run = wandb.init(**self.wandb_params["init"], config=self.wandb_params["config"],
                                  settings=wandb.Settings(start_method="fork"))
        elif wandb_log:
            self.run = run

    def _check_log_path(self):
        log_conf = self.log_config
        exp_folder = '.' if log_conf['exp_folder'] is None else log_conf['exp_folder']
        log_folder = '/data' if log_conf['log_folder'] is None else log_conf['log_folder']
        log_path = check_dir(exp_folder + log_folder)
        return log_path

    def log(self, file_name, log_dict, key_func=None):  # stats, em_iter, ident):
        log_path = self._check_log_path()
        self._log(log_path, file_name, log_dict, key_func)

    def _log(self, log_path, file_name, log_dict, key_func=None):
        if key_func is not None:
            np.savez_compressed(log_path + '/' + file_name, **{key_func(key): log_dict[key] for key in log_dict.keys()})
        else:
            np.savez_compressed(log_path + '/' + file_name, **log_dict)

    def emname(self, em_iter, ident):
        return "logs_em=%d_ident=%s" % (int(em_iter), ident)

    def report(self, log_prob, preds=None, transmat=None, startprob=None, means=None, covars=None, omega_gt=None,
               learned_omega=None, z=None, z0=None, u=None, loss=0):
        super(HMMLoggingMonitor, self).report(log_prob)

        self.z.append(z)
        self.z0.append(z0)
        self.u.append(u)
        self.loss.append(loss)

        if (learned_omega is not None) & (omega_gt is not None):
            omega_diff = dtv(learned_omega, omega_gt)
        else:
            omega_diff = None
        self.omega_dtv.append(omega_diff)
        if self.wandb_log:
            acc = None
            transmat_dtv = None
            startprob_dtv = None
            means_mae = None
            covars_mae = None
            # provide metrics

            if (self.true_states is None) | (preds is None):
                self.run.log({
                    "total_log_prob": log_prob,
                    "accuracy": None,
                    "time": time.perf_counter() - self._init_time,
                    "transmat_dtv": None,
                    "startprob_dtv": None,
                    "means_mae": None,
                    "covars_mae": None,
                    "omage_dtv": omega_diff
                })
            else:
                perm = find_permutation(preds, self.true_states)

                acc = (self.true_states == np.array([perm[i] for i in preds])).mean() if preds is not None else None
                transmat_dtv = dtv(transmat, self.true_transmat[perm, :][:, perm]) if (transmat is not None) & (
                            self.true_transmat is not None) else None
                startprob_dtv = dtv(startprob, self.true_startprob[perm]) if (startprob is not None) & (
                            self.true_startprob is not None) else None
                means_mae = (abs(self.true_means[perm] - means)).mean() if (means is not None) & (
                            self.true_means is not None) else None
                covars_mae = (abs(self.true_covars[perm] - covars)).mean() if (covars is not None) & (
                            self.true_covars is not None) else None

                self.run.log({
                    "total_log_prob": log_prob,
                    "accuracy": acc,
                    "time": time.perf_counter() - self._init_time,
                    "transmat_dtv": transmat_dtv,
                    "startprob_dtv": startprob_dtv,
                    "means_mae": means_mae,
                    "covars_mae": covars_mae,
                    "omage_dtv": omega_diff
                })
            self.accuracy.append(acc)


class DenseHMMLoggingMonitor(HMMLoggingMonitor):
    @property
    def converged(self):
        """Whether the Co-oc algorithm converged."""
        # XXX we might want to check that ``log_prob`` is non-decreasing.
        return (self.iter == self.n_iter or
                (len(self.omega_dtv) >= 2 and
                 list(self.omega_dtv)[-2] - list(self.omega_dtv)[-1] < self.tol))


class GammaGaussianHMM(GaussianHMM):
    """ Base class for Hidden Markov Model of hmmlearn extended for computing all gamma values and logging """

    def __init__(self, n_hidden_states=1, n_dims=None,
                 covariance_type='full', min_covar=0.001,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0, covars_prior=0.01, covars_weight=1,
                 random_state=None, em_iter=10, convergence_tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc", logging_monitor=None, early_stopping=True):

        self.matrix_initializer = None
        if init_params is None:
            init_params = "stmc"
        elif callable(init_params):
            self.matrix_initializer = init_params
            init_params = ''

        super(GammaGaussianHMM, self).__init__(n_components=n_hidden_states,
                                               covariance_type=covariance_type,
                                               min_covar=min_covar,
                                               startprob_prior=startprob_prior,
                                               transmat_prior=transmat_prior,
                                               means_prior=means_prior,
                                               means_weight=means_weight,
                                               covars_prior=covars_prior,
                                               covars_weight=covars_weight,
                                               algorithm="viterbi",
                                               random_state=random_state,
                                               n_iter=em_iter,
                                               tol=convergence_tol,
                                               verbose=verbose,
                                               params=params,
                                               init_params=init_params,
                                               implementation='log')
        self.convergence_tol = convergence_tol
        self.em_iter = em_iter
        self.n_hidden_states = n_hidden_states
        self.logging_monitor = logging_monitor if logging_monitor is not None else HMMLoggingMonitor(
            tol=convergence_tol,
            n_iter=0,
            verbose=verbose)
        self.early_stopping = early_stopping
        if self.matrix_initializer is not None:
            self._init_matrices_using_initializer(self.matrix_initializer)

    def _init_gammas(self, n_seqs, max_seqlen):
        gamma = np.zeros((n_seqs, max_seqlen, self.n_components))
        bar_gamma = np.zeros((max_seqlen, self.n_components))
        gamma_pairwise = np.zeros((n_seqs, max_seqlen - 1, self.n_components, self.n_components))
        bar_gamma_pairwise = np.zeros((max_seqlen - 1, self.n_components, self.n_components))
        return gamma, bar_gamma, gamma_pairwise, bar_gamma_pairwise

    def _initialize_sufficient_statistics(self, n_seqs, max_seqlen):
        """ Initialize a dictionary holding:

            - nobs: int; Number of samples in the data processed so far
            - start: array, shape (n_hidden_states,);
                     An array where the i-th element corresponds to the posterior
                     probability of the first sample being generated by the i-th
                     state.
            - trans: array, shape (n_hidden_states, n_hidden_states);
            An array where the (i, j)-th element corresponds to the
            posterior probability of transitioning between the i-th to j-th
            states.
            - obs: array, shape (n_hidden_states, n_features);
            - obs**2: array, shape (n_hidden_states, n_features);
            An array where the i-th row corresponds to the values of observations and their squares
            - obs*obs.T:  array, shape (n_components, n_features, n_features)
            An array of  covariance matrices for each hidden state

            - max_seqlen: int; Maximum sequence length in given data
            - n_seqs: int; Number of sequences in given data
            - gamma: array, shape (n_seqs, max_seqlen, n_hidden_states);
            The posterior probabilities w.r.t. every observation sequence
            - bar_gamma: array, shape (n_seqs, max_seqlen, n_hidden_states);
            Cumulative posterior probabilities (Sum of gamma over first dimension)
            - gamma_pairwise: array, shape (n_seqs, max_seqlen-1, n_hidden_states, n_hidden_states);
            Pairwise gamma terms over all observation sequences and times
            - bar_gamma_pairwise: array, shape (max_seqlen-1, n_hidden_states, n_hidden_states);
            Cumulative pairwise gamma terms (Sum of gamma_pairwise over first dim.)

            This function is called before every em-iteration.     """

        stats = super(GammaGaussianHMM, self)._initialize_sufficient_statistics()

        stats['max_seqlen'] = max_seqlen
        stats['n_seqs'] = n_seqs

        stats['gamma'], stats['bar_gamma'], stats['gamma_pairwise'], stats['bar_gamma_pairwise'] = self._init_gammas(
            n_seqs, max_seqlen)

        stats['all_logprobs'] = np.zeros((n_seqs,))

        return stats

    """ Initializes the n_features (number of observable dimensions),
        checks input data for Normal distribution and
        initializes Matrices
    """

    def _init(self, X, lengths=None):

        X, n_seqs, max_seqlen = check_arr_gaussian(X, lengths)

        # This initializes the transition matrices:
        # startprob_ and transmat_ to 1/n_hidden_states
        # emissionprob_ to randomly chosen distributions
        # and sets self.n_features to the number of unique symbols in X
        # and checks that X are samples of a multinomial distribution
        # super(GammaMultinomialHMM, self)._init(X, lengths=lengths)
        super(GammaGaussianHMM, self)._init(X)

        if self.matrix_initializer is not None:
            self._init_matrices_using_initializer(self.matrix_initializer)

        return X, n_seqs, max_seqlen

    def _init_matrices_using_initializer(self, matrix_initializer):
        # If random_state is None, this returns a new np.RandomState instance
        # If random_state is int, a new np.RandomState instance seeded with random_state
        # is returned. If random_state is already an instance of np.RandomState,
        # it is returned
        self.random_state = check_random_state(self.random_state)

        pi, A = matrix_initializer(self.n_components, self.n_features, self.random_state)
        self.startprob_, self.transmat_ = pi, A
        self.means_, self.covars_ = 0, 0  # TODO: co to  za  śmiecie?
        self._check()

    def fit(self, X, lengths=None, val=None, val_lengths=None):
        """Estimate model parameters.
        An initialization step is performed before entering the
        EM algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        Returns
        -------
        self : object
            Returns self.
        """

        # Initializes the n_dims (number of dimensions),
        # checks input data for Multinomial distribution and
        # initializes matrices
        X, n_seqs, max_seqlen = self._init(X, lengths=lengths)

        # This makes sure that transition matrices have the correct shape
        # and represent distributions
        self._check()  # INFO: tutaj jest zapisane n_features

        log_config = self.logging_monitor.log_config
        self.logging_monitor._reset()
        for iter in tqdm(range(self.n_iter), desc="Fit model"):

            stats = self._initialize_sufficient_statistics(n_seqs, max_seqlen)  # TODO:  porównaj
            stats["iter"] = iter

            # Do E-step
            stats, total_logprob = self._forward_backward_gamma_pass(X, lengths, stats)

            self._do_mstep(stats)

            self.logging_monitor.report(total_logprob, preds=self.predict(X, lengths), transmat=self.transmat_,
                                        startprob=self.startprob_, means=self.means_, covars=self._covars_)  # TODO: check how to add self covars
            if self.logging_monitor.converged and self.early_stopping:
                print("Exiting EM early ... (convergence tol)")
                break

        return self

    def _forward_backward_gamma_pass(self, X, lengths=None, stats=None, params=None):
        """
        This  is mainly the E step of EM algorithm. TODO: why is it custom?
        Args:
            X:
            lengths:
            stats:
            params:

        Returns:

        """
        if stats is None:
            X, n_seqs, max_seqlen = check_arr_gaussian(X, lengths)
            stats = self._initialize_sufficient_statistics(n_seqs, max_seqlen)

        total_logprob = 0

        for seq_idx, (i, j) in enumerate(iter_from_Xlengths(X, lengths)):
            sub_X = X[i:j]
            lattice, log_prob, posteriors, fwdlattice, bwdlattice = \
                self._fit_log(sub_X)
            # Derived HMM classes will implement the following method to
            # update their probability distributions, so keep
            # a single call to this method for simplicity.
            self._accumulate_sufficient_statistics(
                stats, sub_X, lattice, posteriors, fwdlattice,
                bwdlattice)
            total_logprob += log_prob

            posteriors = pad_to_seqlen(posteriors, stats['max_seqlen'])

            # Compute pairwise gammas and log_xi_sum
            cur_gamma_pairwise = np.zeros_like(stats['bar_gamma_pairwise'])

            # Compute gammas
            stats['gamma'][seq_idx, :, :] = posteriors
            stats['bar_gamma'] += posteriors
            stats['bar_gamma_pairwise'] += cur_gamma_pairwise
            stats['gamma_pairwise'][seq_idx, :, :, :] = cur_gamma_pairwise

        return stats, total_logprob

    # Currently supported: Total log-likelihood on given data,
    # individual log-likelihoods
    def _compute_metrics(self, X, lengths, stats, em_iter, ident,
                         val=None, val_lengths=None):

        log_config = self.logging_monitor.log_config
        log_dict = {}
        stats = stats.copy()

        # log_dict shall contain:
        # After E and M:
        # loglike, all_loglikes
        # val_loglike, val_all_loglikes
        # gamma, bar_gamma, gamma_pairwise, bar_gamma_pairwise
        # startprob_, transmat_, emissionprob_
        # If parameter is set, after E and M: samples
        # If parameter is set, after E:
        # val_gamma, val_bar_gamma, val_gamma_pairwise, val_bar_gamma_pairwise
        # If parameter is set, after M:
        # val_gamma, val_bar_gamma, val_gamma_pairwise, val_bar_gamma_pairwise

        # Gamma and transition matrices
        log_dict['startprob'], log_dict['transmat'], log_dict['mu'], log_dict[
            'sigma'] = self.startprob_, self.transmat_, self.means_, self._covars_

        # Get log-likelihoods on training set
        if ident == 'aE':

            if log_config['gamma_after_estep']:
                log_dict['gamma'], log_dict['bar_gamma'] = stats['gamma'], stats['bar_gamma']
                log_dict['gamma_pairwise'], log_dict['bar_gamma_pairwise'] = stats['gamma_pairwise'], stats[
                    'bar_gamma_pairwise']

            # After the estep, you can get the current log-likelihoods from the stats dict
            log_dict['loglike'] = np.sum(stats['all_logprobs'])
            log_dict['all_loglikes'] = stats['all_logprobs']

            if log_config['samples_after_estep'] is not None:
                sample_sizes = None
                if type(log_config['samples_after_estep']) == tuple:
                    sample_sizes = log_config['samples_after_estep']
                else:
                    if val_lengths is not None:
                        sample_sizes = (len(val_lengths), np.max(val_lengths))
                    else:
                        sample_sizes = (stats['n_seqs'], stats['max_seqlen'])
                log_dict['samples'] = self.sample_sequences(*sample_sizes)

        elif ident == 'aM' or ident == 'f' or ident == 'i':

            # After the mstep, the stats dict contains the log-likelihoods under previous transition matrices
            # Therefore, cannot use stats dict
            log_dict['all_loglikes'] = self.score_individual_sequences(X, lengths)[0]
            log_dict['loglike'] = np.sum(log_dict['all_loglikes'])  # self.score(X, lengths)

            if log_config['gamma_after_mstep']:
                log_dict['gamma'], log_dict['bar_gamma'] = stats['gamma'], stats['bar_gamma']
                log_dict['gamma_pairwise'], log_dict['bar_gamma_pairwise'] = stats['gamma_pairwise'], stats[
                    'bar_gamma_pairwise']

            if log_config['samples_after_mstep'] is not None:
                sample_sizes = None
                if type(log_config['samples_after_mstep']) == tuple:
                    sample_sizes = log_config['samples_after_mstep']
                else:
                    if val_lengths is not None:
                        sample_sizes = (len(val_lengths), np.max(val_lengths))
                    else:
                        sample_sizes = (stats['n_seqs'], stats['max_seqlen'])
                log_dict['samples'] = self.sample_sequences(*sample_sizes)

        # Get log-likelihoods and gammas on test set
        if val is not None:

            if log_config['test_gamma_after_estep'] and ident == 'aE' or log_config[
                'test_gamma_after_mstep'] and ident == 'aM':

                val_stats, val_loglike = self._forward_backward_gamma_pass(val, val_lengths)
                log_dict['val_all_loglikes'] = val_stats['all_logprobs']
                log_dict['val_loglike'] = val_loglike

                # Gammas
                log_dict['val_gamma'], log_dict['val_bar_gamma'] = val_stats['gamma'], val_stats['bar_gamma']
                log_dict['val_gamma_pairwise'] = val_stats['val_gamma_pairwise']
                log_dict['val_bar_gamma_pairwise'] = val_stats['val_bar_gamma_pairwise']

            else:
                log_dict['val_all_loglikes'], log_dict['val_loglike'] = self.score_individual_sequences(val,
                                                                                                        val_lengths)

        return log_dict

    """ Sample n_seqs (int) sequences each of length seqlen (int). Returns an array of shape (n_seqs, seqlen, n_dims) """

    def sample_sequences(self, n_seqs, seqlen):
        return np.array([self.sample(n_samples=seqlen)[0] for _ in range(n_seqs)])

    def score_individual_sequences(self, X, lengths=None):
        """Compute the log probability under the model.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_dims)
            Feature matrix of individual samples.
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.
        Returns
        -------
        logprob : float
            Log likelihood of ``X``.
        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """
        check_is_fitted(self, "startprob_")
        self._check()

        logprobs = None
        if lengths is None:
            logprobs = np.zeros(1)
        else:
            logprobs = np.zeros(len(lengths))

        X = check_array(X)
        # XXX we can unroll forward pass for speed and memory efficiency.
        logprob = 0
        for seq_idx, (i, j) in enumerate(iter_from_Xlengths(X, lengths)):
            framelogprob = self._compute_log_likelihood(X[i:j])
            logprobij, _fwdlattice = self._do_forward_log_pass(framelogprob)
            logprobs[seq_idx] = logprobij
            logprob += logprobij
        return logprobs, logprob

    """ Turns the given observations X of shape (n_sequences, 1)
        into an observations matrix (n_sequences, max_seqlen) 
        by padding sequences to max_seqlen """


class StandardGaussianHMM(GammaGaussianHMM):

    def __init__(self, n_hidden_states=1, n_dims=None,
                 covariance_type='full', min_covar=0.001,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0, covars_prior=0.01, covars_weight=1,
                 random_state=None, em_iter=10, convergence_tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc", logging_monitor=None,
                 early_stopping=True):

        super(StandardGaussianHMM, self).__init__(n_hidden_states=n_hidden_states,
                                                  n_dims=n_dims,
                                                  covariance_type=covariance_type,
                                                  min_covar=min_covar,
                                                  startprob_prior=startprob_prior,
                                                  transmat_prior=transmat_prior,
                                                  means_prior=means_prior,
                                                  means_weight=means_weight,
                                                  covars_prior=covars_prior,
                                                  covars_weight=covars_weight,
                                                  random_state=random_state,
                                                  em_iter=em_iter,
                                                  convergence_tol=convergence_tol,
                                                  verbose=verbose,
                                                  params=params,
                                                  init_params=init_params,
                                                  logging_monitor=logging_monitor,
                                                  early_stopping=early_stopping)

    """ Computes loss on given sequence; Using given gamma terms and
    the current transition matrices
    """

    def _compute_loss(self, X, lengths, bar_gamma, bar_gamma_pairwise, gamma):
        log_A = np.log(self.transmat_)
        log_B = np.log(np.array(
            [
                [
                    [
                        multivariate_normal.pdf(x, m, c)
                        for m, c in zip(self.means_, self._covars_)
                    ]
                    for x in X[i:j]
                ] for seq_idx, (i, j) in enumerate(iter_from_Xlengths(X, lengths))
            ]
        ))
        log_pi = np.log(self.startprob_)

        loss1 = -np.einsum('s,s->', log_pi, bar_gamma[0, :])
        loss2 = -np.einsum('jl,tjl->', log_A, bar_gamma_pairwise)
        # loss3 = -np.einsum('jit,itj->', log_B, gamma)
        loss3 = -np.einsum('ijt,ijt->', log_B, gamma)  # TODO:  check!
        loss = loss1 + loss2 + loss3

        return np.array([loss, loss1, loss2, loss3])


class GaussianDenseHMM(GammaGaussianHMM):
    SUPPORTED_REPRESENTATIONS = frozenset({'uzz0mc'})
    SUPPORTED_OPT_SCHEMES = frozenset(('em', 'cooc'))

    def __init__(self, n_hidden_states=1, n_dims=None,
                 covariance_type='full', min_covar=0.001,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0, covars_prior=0.01, covars_weight=1,
                 random_state=None, em_iter=10, convergence_tol=1e-2, verbose=False,
                 params="stmc", init_params="stmc", logging_monitor=None,
                 mstep_config=None, opt_schemes=None, early_stopping=True, nodes=None):

        super(GaussianDenseHMM, self).__init__(n_hidden_states=n_hidden_states,
                                               n_dims=n_dims,
                                               covariance_type=covariance_type,
                                               min_covar=min_covar,
                                               startprob_prior=startprob_prior,
                                               transmat_prior=transmat_prior,
                                               means_prior=means_prior,
                                               means_weight=means_weight,
                                               covars_prior=covars_prior,
                                               covars_weight=covars_weight,
                                               random_state=random_state,
                                               em_iter=em_iter,
                                               convergence_tol=convergence_tol,
                                               verbose=verbose,
                                               params=params,
                                               init_params=init_params,
                                               logging_monitor=logging_monitor,
                                               early_stopping=early_stopping)

        mstep_config = {} if mstep_config is None else mstep_config
        self.mstep_config = mstep_config
        self.opt_schemes = self.SUPPORTED_OPT_SCHEMES if opt_schemes is None else set(opt_schemes)

        # Used for both optimization schemes
        self.initializer = dict_get(mstep_config, 'initializer', default=tf.initializers.RandomNormal)
        self.l_uz = dict_get(mstep_config, 'l_uz', default=3)
        self.trainables = dict_get(mstep_config, 'trainables', default='uzz0mc')
        self.representations = dict_get(mstep_config, 'representations', default='uzz0mc')
        self.kernel = dict_get(mstep_config, 'kernel', default='exp')

        # TF variables and placeholders
        self.u, self.z, self.z0 = None, None, None  # Representations

        # Only needed for EM optimization
        self.em_epochs = dict_get(mstep_config, 'em_epochs', default=10)
        self.em_lr = dict_get(mstep_config, 'em_lr', default=0.01)
        self.em_optimizer = dict_get(mstep_config, 'em_optimizer', default=None)
        self.scaling = dict_get(mstep_config, 'scaling', default=n_hidden_states)

        # Only needed for cooc optimization
        self.loss_type = dict_get(mstep_config, 'loss_type', default='abs_log')
        self.cooc_lr = dict_get(mstep_config, 'cooc_lr', default=0.001)
        self.cooc_optimizer = dict_get(mstep_config, 'cooc_optimizer',
                                       default=None)
        self.cooc_epochs = dict_get(mstep_config, 'cooc_epochs', default=10)
        self.discrete_nodes = check_nodes(nodes) if nodes is not None else None
        self.splits = None
        self.discrete_observables = None

    def compute_stationary(self, M, verbose=True):
        eigval, eigvec = tf.linalg.eig(tf.transpose(M))
        idx = tf.experimental.numpy.nonzero(tf.experimental.numpy.isclose(eigval, [1.]))[0]
        # if idx.size < 1:  # TODO: fix and enable
        #     raise Exception("No Eigenvalue 1")
        # elif idx.size > 1 and verbose:
        #     print("Warning: Multiple vectors corresponding to eigenvalue 1.: %s" % str(idx))
        M_stationary = tf.math.real(eigvec[:, idx[0]])
        M_stationary = M_stationary / tf.math.reduce_sum(M_stationary)
        return M_stationary

    def calculate_all_scalars(self, recover_B=True):
        """ Recovering A, B, pi """
        # Compute scalar products
        A_scalars = tf.matmul(self.u, self.z, name="A_scalars")
        pi_scalars = tf.matmul(self.u, self.z0, name="pi_scalars")

        # Apply kernel
        A_from_reps = tf.math.softmax(A_scalars, axis=0)
        pi_from_reps = tf.math.softmax(pi_scalars, axis=0)

        # hmmlearn library uses a different convention for the shapes of the matrices
        A_from_reps_hmmlearn = tf.transpose(a=A_from_reps, name='A_from_reps')
        pi_from_reps_hmmlearn = tf.reshape(pi_from_reps, (-1,), name='pi_from_reps')

        if recover_B:

            if self.covariance_type == "full":
                L = tfp.math.fill_triangular(self.covars_vec, name="L")
                covars_cooc = tf.matmul(L, tf.transpose(a=L, perm=[0, 2, 1]), name="covars_direct")
            elif self.covariance_type == "diag":
                covars_cooc = tf.linalg.diag(self.covars_vec)

            if self.n_features == 1:
                B_scalars_tmp = .5 * (
                        1 + tf.math.erf((self.discrete_nodes - tf.transpose(a=self.means_cooc)) / (
                        tf.nn.relu(tf.transpose(a=covars_cooc[:, :, 0])) + 1e-10) / np.sqrt(2)))

                B_scalars = tf.transpose(a=B_scalars_tmp[1:, :] - B_scalars_tmp[:-1, :], name="B_scalars_cooc")
            elif self.n_features == 2:
                B_scalars_tmp = None
                if self.covariance_type == "full":
                    def map_B(n):
                        res = tf.reduce_mean(
                            input_tensor=tf.cast(tf.reduce_all(input_tensor=mvn_sample <= n, axis=-1), mvn.dtype),
                            axis=0)
                        return res
                    mvn = tfp.distributions.MultivariateNormalTriL(self.means_cooc, covars_cooc)
                    mvn_sample = mvn.sample(100000, seed=2022)
                    B_scalars_tmp = tf.map_fn(map_B, self.discrete_nodes)
                elif self.covariance_type == "diag":
                    B_scalars_tmp = tf.reduce_prod(input_tensor=.5 * (
                            1 + tf.math.erf((tf.expand_dims(self.discrete_nodes, axis=-1) - tf.expand_dims(
                        tf.transpose(a=self.means_cooc), axis=0)) / (
                                                    tf.nn.relu(tf.expand_dims(tf.transpose(a=self.covars_vec),
                                                                              axis=0)) + 1e-10) / np.sqrt(2))), axis=1)
                B_scalars_tmp_wide = tf.reshape(B_scalars_tmp, (*[n.shape[0] for n in self.splits], self.n_components))
                B_scalars = tf.transpose(a=tf.reshape(
                    B_scalars_tmp_wide[:-1, 1:, :] - B_scalars_tmp_wide[:-1, :-1, :] - B_scalars_tmp_wide[1:, 1:, :] + B_scalars_tmp_wide[1:, :-1, :],
                    (-1, self.n_components)), name="B_scalars_cooc")
            else:
                raise Exception("Co-occurrences for dimensionality >=  3  not implemented.")
        else:
            B_scalars, covars_cooc = None, None

        return A_from_reps_hmmlearn, pi_from_reps_hmmlearn, B_scalars, covars_cooc

    @tf.function
    def em_loss_update(self):  # TODO: sprawdź argumenty, te które są w selfie czytaj bezpośrednio z selfa.
        X = self.X
        # O = self.O
        gamma, bar_gamma, bar_gamma_pairwise = self.gamma, self.bar_gamma, self.bar_gamma_pairwise



        """ Recovering A, B, pi """
        # Compute scalar products
        A_scalars = tf.matmul(self.u, self.z, name="A_scalars")
        pi_scalars = tf.matmul(self.u, self.z0, name="pi_scalars")

        # Apply kernel
        A_from_reps = tf.math.softmax(A_scalars, axis=0)  # self.A_from_reps_hmmlearn, self.pi_from_reps_hmmlearn
        pi_from_reps = tf.math.softmax(pi_scalars, axis=0)

        A_log_ker_normal = tf.reduce_logsumexp(A_scalars, axis=0)  # L
        pi_log_ker_normal = tf.reduce_logsumexp(pi_scalars)  # L0

        A_log_ker = tf.identity(A_scalars, name='A_log_ker')
        pi_log_ker = tf.identity(pi_scalars, name='pi_log_ker')

        mvn = tfp.distributions.MultivariateNormalTriL(self.means_, self.covars_)
        B_scalars = tf.map_fn(mvn.prob, X, name="B_scalars_em")
        if self.kernel == 'exp' or self.kernel == tf.exp:
            B_log_ker = tf.math.log(B_scalars, name='B_log_ker_em')
        else:
            B_scalars_ker = B_scalars
            B_log_ker = tf.math.log(B_scalars_ker, name='B_log_ker_em')

        # Losses
        bar_gamma_1 = bar_gamma[0, :]
        loss_1 = -tf.reduce_sum(input_tensor=pi_log_ker * bar_gamma_1, name="loss_1")
        loss_1_normalization = tf.reduce_sum(input_tensor=pi_log_ker_normal * bar_gamma_1, name="loss_1_normalization")
        loss_2 = -tf.reduce_sum(input_tensor=A_log_ker * bar_gamma_pairwise, name="loss_2")
        loss_2_normalization = tf.reduce_sum(
            input_tensor=A_log_ker_normal[tf.newaxis, :, tf.newaxis] * bar_gamma_pairwise,
            name="loss_2_normalization")
        # tilde_M = tf.einsum('ito,oh->ith', np.eye(1)[O], B_log_ker)
        # loss_3 = -tf.reduce_sum(gamma * tilde_M)
        loss_3 = -tf.reduce_sum(input_tensor=gamma.reshape(-1, self.n_components)[:B_log_ker.shape[0], :] * B_log_ker, name="loss_3")

        loss_total = tf.identity(loss_1 + loss_1_normalization +
                                 loss_2 + loss_2_normalization +
                                 loss_3,
                                 name="loss_total")

        loss_scaled = tf.identity(loss_total / self.scaling, name="loss_scaled")

        return loss_scaled


    @tf.function
    def cooc_loss_update(self):  #  TODO:  get A< A_stationary...

        A_from_reps_hmmlearn, pi_from_reps_hmmlearn, B_scalars, covars_cooc = self.calculate_all_scalars()

        A_stationary = self.compute_stationary(A_from_reps_hmmlearn, verbose=False)
        omega_gt = self.omega_gt

        theta = A_from_reps_hmmlearn * A_stationary[:, None]
        omega = tf.matmul(tf.transpose(a=B_scalars), tf.matmul(theta, B_scalars))
        self.omega = omega
        if self.loss_type == "square":
            loss_cooc = tf.reduce_sum(input_tensor=tf.square(omega_gt - omega))
        elif self.loss_type == "square_log":
            loss_cooc = tf.reduce_sum(input_tensor=tf.square(tf.math.log(omega_gt) - tf.math.log(omega)))
        else:
            loss_cooc = tf.reduce_sum(input_tensor=tf.math.abs(tf.math.log(omega_gt) - tf.math.log(omega)))

        self.loss_cooc = loss_cooc
        return loss_cooc


    def _init_tf(self):  # TODO:  rewrite all!
        if self.representations not in self.SUPPORTED_REPRESENTATIONS:  # TODO: remove all representation stuff!!
            raise Exception("Given representation argument is invalid. Has to be one of %s" %
                            str(self.SUPPORTED_REPRESENTATIONS))

        if len(self.opt_schemes.difference(self.SUPPORTED_OPT_SCHEMES)) > 0:
            raise Exception(
                "Given unsupported optimization scheme! Supported are: %s" % str(self.SUPPORTED_OPT_SCHEMES))

        # Trainables in both fit methods
        u = tf.Variable(name="u", dtype=tf.float64, shape=[self.n_components, self.l_uz],
                        initial_value=self.initializer(0., 1.)(shape=(self.n_components, self.l_uz)).astype('float64'),
                        trainable=('u' in self.trainables))

        z = tf.Variable(name="z", dtype=tf.float64, shape=[self.l_uz, self.n_components],
                        initial_value=self.initializer(0., 1.)(shape=(self.l_uz, self.n_components)).astype('float64'),
                        trainable=('z' in self.trainables and
                                   ('z0' not in self.trainables
                                    or 'zz0' in self.trainables)))
        z0 = tf.Variable(name="z0", dtype=tf.float64, shape=[self.l_uz, 1],
                         initial_value=self.initializer(0., 1.)(shape=(self.l_uz, 1)).astype('float64'),
                         trainable=('z0' in self.trainables))

        self.u, self.z, self.z0 = u, z, z0

        if 'cooc' in self.opt_schemes:
            # Additional trainables in fit_cooc
            means_cooc = tf.Variable(name="means_cooc", dtype=tf.float64,
                                     shape=[self.n_components, self.n_features],
                                     initial_value=self.means_.astype('float64'),
                                     trainable=('m' in self.trainables))
            if self.covariance_type == "full":
                init_val = np.triu(self._covars_)
                init_val = np.transpose(init_val[init_val != 0]).reshape(self.n_components,
                                                                         (self.n_features * (self.n_features + 1)) // 2)
                covars_vec = tf.Variable(name="covars_cooc", dtype=tf.float64,
                                         shape=[self.n_components, (self.n_features * (self.n_features + 1)) // 2],
                                         initial_value=init_val.astype('float64'),
                                         trainable=('c' in self.trainables))


            elif self.covariance_type == "diag":
                covars_vec = tf.Variable(name="covars_cooc", dtype=tf.float64,
                                         shape=[self.n_components, self.n_features],
                                         initial_value=self._covars_.astype('float64'),
                                         trainable=('c' in self.trainables))

            self.means_cooc = means_cooc
            self.covars_vec = covars_vec
            
    
    def _init(self, X, lengths=None):
        X, n_seqs, max_seqlen = check_arr_gaussian(X, lengths)
        self.X = X
        # O, n_seqs, max_seqlen = self._observations_to_padded_matrix(X, lengths)
        # self.X = O

        if self.matrix_initializer is not None:
            self._init_matrices_using_initializer(self.matrix_initializer)

        _check_and_set_gaussian_n_features(self, X)
        super(GaussianHMM, self)._init(X)
        kmeans = cluster.KMeans(n_clusters=self.n_components,
                                random_state=self.random_state)
        kmeans.fit(X)
        if self._needs_init("m", "means_"):
            self.means_ = kmeans.cluster_centers_

        if self.discrete_nodes is None:
            dtree = DecisionTreeClassifier(max_depth=np.ceil(np.log2(self.n_components)).astype(int)).fit(X, kmeans.labels_)
            self._init_nodes(X, dtree)

        if self._needs_init("c", "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self.covars_ = \
                _utils.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components).copy()

        self._init_tf()

        return X, n_seqs, max_seqlen

    """ Learns representations, recovers transition matrices and sets them """

    def _do_mstep(self, stats):

        # Optimizer step
        if self.em_optimizer is None:
            self.em_optimizer = tf.keras.optimizers.Adam(learning_rate=self.em_lr, name="adam_em")

        it = stats["iter"]
        self.gamma = stats['gamma']
        self.bar_gamma = stats['bar_gamma']
        self.bar_gamma_pairwise = stats['bar_gamma_pairwise']

        if self.em_optimizer is None:
            self.em_optimizer = tf.keras.optimizers.Adam(learning_rate=self.em_lr, name="adam_em")

        for epoch in range(self.em_epochs):
            self.em_optimizer.minimize(self.em_loss_update,
                                         var_list=[self.u, self.z, self.z0],
                                         tape=tf.GradientTape())
            if self.verbose:
                cur_loss = tf.get_static_value(self.em_loss_update())
                print("Loss at epoch %d is %.8f" % (
                    epoch, cur_loss))

        # A, pi = self.A_from_reps_hmmlearn.numpy(), self.pi_from_reps_hmmlearn.numpy()  # TODO
        A, pi, _, _ = self.calculate_all_scalars(recover_B=False)
        self.transmat_ = A
        self.startprob_ = pi

        # Update means and covars like in GaussianHMM

        means_prior = self.means_prior
        means_weight = self.means_weight

        denom = stats['post'][:, None]
        if 'm' in self.params:  # INFO: Update parameters only if new value is defined
            vals_tmp = ((means_weight * means_prior + stats['obs'])
                        / (means_weight + denom))
            if np.isnan(vals_tmp).sum() == 0:
                self.means_ = vals_tmp

        if 'c' in self.params:
            covars_prior = self.covars_prior
            covars_weight = self.covars_weight
            meandiff = self.means_ - means_prior

            if self.covariance_type in ('spherical', 'diag'):
                c_n = (means_weight * meandiff ** 2
                       + stats['obs**2']
                       - 2 * self.means_ * stats['obs']
                       + self.means_ ** 2 * denom)
                c_d = max(covars_weight - 1, 0) + denom
                vals_tmp = (covars_prior + c_n) / np.maximum(c_d, 1e-5)
                if np.isnan(vals_tmp).sum() == 0:
                    self._covars_ = vals_tmp
                if self.covariance_type == 'spherical':
                    vals_tmp = np.tile(self._covars_.mean(1)[:, None],
                                       (1, self._covars_.shape[1]))
                    if np.isnan(vals_tmp).sum() == 0:
                        self._covars_ = vals_tmp
            elif self.covariance_type in ('tied', 'full'):
                c_n = np.empty((self.n_components, self.n_features,
                                self.n_features))
                for c in range(self.n_components):
                    obsmean = np.outer(stats['obs'][c], self.means_[c])
                    c_n[c] = (means_weight * np.outer(meandiff[c],
                                                      meandiff[c])
                              + stats['obs*obs.T'][c]
                              - obsmean - obsmean.T
                              + np.outer(self.means_[c], self.means_[c])
                              * stats['post'][c])
                cvweight = max(covars_weight - self.n_features, 0)
                if self.covariance_type == 'tied':
                    vals_tmp = ((covars_prior + c_n.sum(axis=0)) /
                                (cvweight + stats['post'].sum()))
                    if np.isnan(vals_tmp).sum() == 0:
                        self._covars_ = vals_tmp
                elif self.covariance_type == 'full':
                    vals_tmp = ((covars_prior + c_n) /
                                (cvweight + stats['post'][:, None, None]))
                    if np.isnan(vals_tmp).sum() == 0:
                        self._covars_ = vals_tmp

    def _compute_metrics(self, X, lengths, stats, em_iter, ident,
                         val=None, val_lengths=None):

        log_config = self.logging_monitor.log_config
        log_dict = super(GaussianDenseHMM, self)._compute_metrics(X, lengths, stats, em_iter, ident, val, val_lengths)

        log_dict['u'], log_dict['z'], log_dict['z_0'] = self.get_representations()
        log_dict['mu'], log_dict['sigma'] = self.means_, self._covars_

        gamma, bar_gamma = stats['gamma'], stats['bar_gamma']
        bar_gamma_pairwise = stats['bar_gamma_pairwise']
        log_dict['train_losses'] = self._compute_loss(X, lengths, bar_gamma, bar_gamma_pairwise, gamma)
        log_dict['train_losses_standard'] = self._compute_loss_standard(X, lengths, bar_gamma, bar_gamma_pairwise,
                                                                        gamma)

        if val is not None:

            log_dict['test_losses'] = self._compute_loss(val, val_lengths, bar_gamma, bar_gamma_pairwise, gamma)
            log_dict['test_losses_standard'] = self._compute_loss_standard(val, val_lengths, bar_gamma,
                                                                           bar_gamma_pairwise, gamma)

            # val-gammas are not necessarily in log_dict
            val_bar_gamma = dict_get(log_dict, 'val_bar_gamma')
            val_bar_gamma_pairwise = dict_get(log_dict, 'val_bar_gamma_pairwise')
            val_gamma = dict_get(log_dict, 'val_gamma')

            if val_bar_gamma is not None and val_bar_gamma_pairwise is not None and val_gamma is not None:
                log_dict['test_gamma_losses'] = self._compute_loss(val, val_lengths, val_bar_gamma,
                                                                   val_bar_gamma_pairwise, val_gamma)
                log_dict['test_gamma_losses_standard'] = self._compute_loss_standard(val, val_lengths, val_bar_gamma,
                                                                                     val_bar_gamma_pairwise, val_gamma)

        return log_dict

    """ Computes same loss as standard hmm """  # TODO:  po co to?

    def _compute_loss_standard(self, X, lengths, bar_gamma, bar_gamma_pairwise, gamma):  # TODO: remove padded  matrix
        log_A = np.log(self.transmat_)
        log_B = np.log(np.array(
            [
                [
                    [
                        multivariate_normal.pdf(x, m, c)
                        for m, c in zip(self.means_, self._covars_)
                    ]
                    for x in X[i:j]
                ] for seq_idx, (i, j) in enumerate(iter_from_Xlengths(X, lengths))
            ]
        ))
        log_pi = np.log(self.startprob_)

        loss1 = -np.einsum('s,s->', log_pi, bar_gamma[0, :])
        loss2 = -np.einsum('jl,tjl->', log_A, bar_gamma_pairwise)
        loss3 = -np.einsum('sti,sti->', log_B, gamma)
        loss = loss1 + loss2 + loss3

        return np.array([loss, loss1, loss2, loss3])

    def _compute_loss(self, X, lengths, bar_gamma, bar_gamma_pairwise, gamma):  #  TODO: rename like 'get current loss'
        losses = [self.loss_1.numpy(), self.loss_1_normalization.numpy(), self.loss_2.numpy(), self.loss_2_normalization.numpy(), self.loss_3.numpy()]
        losses = [np.sum(losses)] + losses
        return np.array(losses)

    """ Fits a GaussianDenseHMM using the co-occurrence optimization scheme 
        If gt_AB = (A, B) is given, X/val are assummed to be generated by a
        stationary HMM with parameters A, B and gt co-occurence is computed analytically

    """

    def fit_coocs(self, X, lengths, val=None, val_lengths=None, gt_AB=None):
        X, n_seqs, max_seqlen = self._init(X, lengths)

        # ic(self._covars_)
        # ic(self.covars_)

        gt_omega = None
        freqs, gt_omega_emp = empirical_coocs(self._to_discrete(X), np.prod(self.discrete_observables), lengths=lengths)
        gt_omega_emp = np.reshape(gt_omega_emp, newshape=(np.prod(self.discrete_observables), np.prod(self.discrete_observables)))

        if gt_AB is not None:
            A, B = gt_AB
            A_stationary = self.compute_stationary(A)
            theta = A * A_stationary[:, None]
            gt_omega = np.matmul(B.T, np.matmul(theta, B))

        gt_omega = gt_omega_emp if gt_omega is None else gt_omega
        self.omega_gt = gt_omega.astype('float64')
        log_dict = self._fit_coocs(X, lengths, val_lengths)

        log_dict['cooc_logprobs'] = self.score_individual_sequences(X, lengths)[0]
        if val is not None and val_lengths is not None:
            log_dict['cooc_val_logprobs'] = self.score_individual_sequences(val, val_lengths)[0]

        self.logging_monitor.log('logs_coocs', log_dict)

    def _init_nodes(self, X, dtree):
        splits = np.concatenate([dtree.tree_.feature.reshape(-1, 1), dtree.tree_.threshold.reshape(-1, 1)], axis=1)

        def _min(x):
            return np.min(x) - 1e-1

        def _max(x):
            return np.max(x) + 1e+1

        splits = np.concatenate([splits, np.array([[i, fun(X[:, i])] for i, fun in itertools.product(range(X.shape[1]), [_min, _max])])])
        splits = splits[splits[:, 0] >= 0]

        nodes_x = [np.sort(splits[splits[:, 0] == float(i), 1]) for i in np.unique(splits[:, 0])]
        nodes = np.array([t for t in itertools.product(*nodes_x)])
        self.splits = nodes_x  # number  of splits  on each axis
        self.discrete_nodes = nodes.astype('float64')
        self.discrete_observables = [n.shape[0] - 1 for n in nodes_x]

    def _to_discrete(self, X):
        indexes = np.arange(np.prod(self.discrete_observables)).reshape(self.discrete_observables)  #.transpose()
        X_disc = np.array([indexes[i] for i in zip(*[(X[:, j].reshape(-1, 1) > self.splits[j].reshape(1, -1)).sum(axis=1) - 1 for j in range(len(self.splits))])])
        return X_disc

    def _fit_coocs(self, X, lengths, val_lengths=None):
        losses = []

        # ic(self._covars_)
        # ic(self.covars_)

        if self.cooc_optimizer is None:
            self.cooc_optimizer = tf.keras.optimizers.Adam(learning_rate=self.cooc_lr, name="adam_cooc")

        for epoch in range(self.cooc_epochs):
            # ic(epoch)
            self.cooc_optimizer.minimize(self.cooc_loss_update,
                                         var_list=[self.u, self.z, self.means_cooc, self.covars_vec],
                                         tape=tf.GradientTape())
            # # ic(self._covars_)
            # # ic(self.covars_)
            if epoch % 100 == 0:
                cur_loss = tf.get_static_value(self.loss_cooc)
                losses.append(cur_loss)  # TODO: can it stay like this??

                A, pi_from_reps_hmmlearn, B_scalars, covars_cooc = self.calculate_all_scalars()
                A_stat = self.compute_stationary(A, verbose=False)
                means_c, covars_c = self.means_cooc.numpy(), tf.get_static_value(
                    tf.matmul(covars_cooc, tf.transpose(covars_cooc, perm=(0, 2, 1))))
                if self.covariance_type == 'diag':
                    covars_c = np.array(list(map(np.diag, covars_c)))
                    # # ic(covars_c)

                # ic(covars_c)

                self.transmat_ = A
                self.means_ = means_c if np.isnan(means_c).sum() == 0 else self.means_
                self._covars_ = covars_c if np.isnan(  # TODO: is the square here needed?
                    covars_c).sum() == 0 else self._covars_  # TODO: fix! It depends on covariance type!
                self.startprob_ = A_stat
                # z, z0, u = self.z.numpy(), self.z0.numpy(), self.u.numpy()
                # self.logging_monitor.report(self.score(X, lengths), # None,  #
                #                             preds=self.predict(X, lengths),
                #                             transmat=A, startprob=A_stat, means=means_c, covars=np.square(covars_c),
                #                             omega_gt=self.omega_gt, learned_omega=tf.get_static_value(omega),
                #                             z=z, z0=z0, u=u, loss=cur_loss)
                self.logging_monitor.report(self.score(X, lengths), loss=cur_loss)
                if self.verbose:
                    print(cur_loss)

        log_dict = {}
        log_dict['cooc_losses'] = losses

        A, pi_from_reps_hmmlearn, B_scalars, covars_cooc = self.calculate_all_scalars()
        A_stat = self.compute_stationary(A, verbose=False)
        theta = A * A_stat[:, None]
        learned_omega = tf.matmul(tf.transpose(a=B_scalars), tf.matmul(theta, B_scalars))

        # ic(covars_cooc)
        # ic(tf.matmul(covars_cooc, tf.transpose(covars_cooc, perm=(0, 2, 1))))
        # ic(tf.get_static_value(tf.matmul(covars_cooc, tf.transpose(covars_cooc, perm=(0, 2, 1)))))

        means_c, covars_c = self.means_cooc.numpy(), tf.get_static_value(tf.matmul(covars_cooc, tf.transpose(covars_cooc, perm=(0, 2, 1))))
        if self.covariance_type == 'diag':
            covars_c = np.array(list(map(np.diag, covars_c)))
            # ic(covars_c)

        # ic(covars_c)

        self.transmat_ = A
        self.means_ = means_c if np.isnan(means_c).sum() == 0 else self.means_
        self._covars_ = covars_c if np.isnan(  # TODO: is the square here needed?
            covars_c).sum() == 0 else self._covars_  # TODO: fix! It depends on covariance type!
        self.startprob_ = A_stat
        self._check()

        log_dict.update({'cooc_transmat': self.transmat_, 'cooc_means': self.means_, 'cooc_covars': self.covars_,
                         'cooc_startprob': self.startprob_, 'cooc_omega': learned_omega})

        u, z = self.u.numpy(), self.z.numpy()
        log_dict.update(dict(u=u, z=z, means=means_c, covars=covars_c))

        if self.logging_monitor.log_config['samples_after_cooc_opt'] is not None:
            sample_sizes = None
            if type(self.logging_monitor.log_config['samples_after_cooc_opt']) == tuple:
                sample_sizes = self.logging_monitor.log_config['samples_after_cooc_opt']
            else:
                if val_lengths is not None:
                    sample_sizes = (len(val_lengths), np.max(val_lengths))
                else:
                    sample_sizes = (len(lengths), np.max(lengths))
            log_dict['cooc_samples'] = self.sample_sequences(*sample_sizes)

        return log_dict

    def get_representations(self):
        return self.u.numpy(), self.z.numpy(), self.z0.numpy()
