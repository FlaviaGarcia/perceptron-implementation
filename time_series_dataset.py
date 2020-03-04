# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 17:44:05 2020

@author: Flavia García Vázquez
"""

import numpy as np 
import matplotlib.pyplot as plt

time_series_map = dict()

def generate_dataset_raw():
    n_points = 1200
    inputs = np.zeros((n_points, 5))
    outputs = np.zeros(n_points)
    for p in range(n_points):
        t = p + 301
        inputs[p][0] = x(t - 20)
        inputs[p][1] = x(t - 15)
        inputs[p][2] = x(t - 10)
        inputs[p][3] = x(t - 5)
        inputs[p][4] = x(t)
        outputs[p] = x(t + 5)
    return inputs, outputs



def generate_dataset():
    inputs, outputs = generate_dataset_raw()
    testing_cutting_index = len(outputs) - 200
    evaluation_cutting_index = testing_cutting_index - 200
    training_inputs = inputs[:evaluation_cutting_index, :]
    training_outputs = outputs[:evaluation_cutting_index]
    validation_inputs = inputs[evaluation_cutting_index:testing_cutting_index, :]
    validation_outputs = outputs[evaluation_cutting_index:testing_cutting_index].reshape(-1,1)
    testing_inputs = inputs[testing_cutting_index:, :]
    testing_outputs = outputs[testing_cutting_index:]
    return training_inputs, training_outputs, validation_inputs, validation_outputs, testing_inputs, testing_outputs, outputs



def x(t):
    if time_series_map.get(t) != None:
        return time_series_map[t]
    if t == 0:
        return 1.5
    elif t < 0:
        return 0
    else:
        num = 0.2 * x(t - 26)
        den = 1 + np.power(x(t - 26), 10)
        value = x(t - 1) + num/den - 0.1 * x(t - 1)
        time_series_map[t] = value
        return value


def plot_generated_data(training_outputs, validation_outputs, testing_outputs,outputs_all, plt_title ="Time series dataset"):
    x_lower_limit_val = len(training_outputs)
    x_upper_limit_val = x_lower_limit_val + len(validation_outputs)
    x_lower_limit_test = x_upper_limit_val 
    x_upper_limit_test = x_lower_limit_test + len(testing_outputs)
    plt.plot(outputs_all, "k", label="All data")
    plt.plot(training_outputs, "--r", label="Train")
    plt.plot(np.arange(x_lower_limit_val, x_upper_limit_val), validation_outputs, "--b",label="Validation")
    plt.plot(np.arange(x_lower_limit_test,x_upper_limit_test), testing_outputs, "--g",label="Test")
    plt.legend()
    plt.title(plt_title)
    plt.show()
    