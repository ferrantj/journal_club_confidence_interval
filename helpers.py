import numpy as np
import scipy.stats as st
import random
from matplotlib import pyplot as plt


def run_many_proportion_trials(true_value, num_trials=50000, num_observations=200):
    observed_values = []
    for _ in range(num_trials):
        trial = [random.random() < true_value for _ in range(num_observations)]
        mean_value = np.mean(trial)
        observed_values.append(mean_value)
    observed_values.sort()
    return observed_values


def display_trials(trials, true_value=None, percentile_bounds=None):
    if true_value:
        plt.axvline(true_value, color='k', linestyle='dashed', linewidth=1, label='True Value')
    if percentile_bounds:
        two_sided = (percentile_bounds + 1.0) / 2
        plt.axvline(trials[int(len(trials) * two_sided)], color='r', linestyle='dashed', linewidth=1, label=f'{int(percentile_bounds*100)}th percentile')
        plt.axvline(trials[-int(len(trials) * two_sided)], color='r', linestyle='dashed', linewidth=1)
    plt.hist(trials)
    plt.legend()
    plt.show()


def run_single_proportion_trial(true_value, num_observations=200):
    trial = [random.random() < true_value for _ in range(num_observations)]
    trial_value = np.mean(trial)
    std_dev = np.std(trial, dtype=np.float64)
    std_err = std_dev/np.sqrt(len(trial))
    return (trial_value, std_err)


def display_single_trial(trial, confidence=None, true_value=None, distribution=None):
    observed_value, std_err = trial
    plt.axvline(observed_value, color='orange', linestyle='dashed', linewidth=1, label='Observed Value')
    if confidence:
        plt.axvline(observed_value + std_err * st.norm.ppf((confidence+1)/2), color='r', linestyle='dashed', linewidth=1, label=f'Confidence Interval')
        plt.axvline(observed_value - std_err * st.norm.ppf((confidence+1)/2), color='r', linestyle='dashed', linewidth=1)
    if true_value:
        plt.axvline(true_value, color='k', linestyle='dashed', linewidth=1, label='True Value')
    if distribution:
        plt.hist(distribution)
    plt.legend()
    plt.show()

def within_confidence_interval(trial, true_sensitivity, confidence):
    mean_value, std_err = trial
    upper_bound = mean_value + std_err * st.norm.ppf((confidence+1)/2)
    lower_bound = mean_value - std_err * st.norm.ppf((confidence+1)/2)
    return (true_sensitivity < upper_bound) and (true_sensitivity > lower_bound)