#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 20:11:05 2020

@author: Flavia García Vázquez
"""

import keras
from keras.regularizers import l2
import time
import time_series_dataset 


def MLP_creation(n_nodes_input, regularization_term, n_hidden_nodes_first_layer, n_hidden_nodes_second_layer=None, n_output_nodes=1, three_layers=False): 

    model = keras.models.Sequential()
    model.add(keras.layers.normalization.BatchNormalization(input_shape=tuple([n_nodes_input])))
    model.add(keras.layers.core.Dense(n_hidden_nodes_first_layer, activation='relu', activity_regularizer=l2(regularization_term)))
    
    if three_layers:
        model.add(keras.layers.normalization.BatchNormalization())
        model.add(keras.layers.core.Dense(n_hidden_nodes_second_layer, activation='relu', activity_regularizer=l2(regularization_term)))


    model.add(keras.layers.core.Dense(n_output_nodes, activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])  

    return model

    

if __name__ == "__main__":
    
    epochs=100000
        
    training_inputs, training_outputs, validation_inputs, validation_outputs, testing_inputs, testing_outputs, outputs_all = time_series_dataset.generate_dataset()
    
    time_series_dataset.plot_generated_data(training_outputs, validation_outputs, testing_outputs, outputs_all)
    
    n_nodes_input = training_inputs.shape[1]
    n_training_samples = training_inputs.shape[0]
    
    model = MLP_creation(n_nodes_input, regularization_term = .0000001, n_hidden_nodes_first_layer=7)
    
    print(model.summary())
    
    callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
    start_time = time.time()
    model.fit(training_inputs, training_outputs, batch_size=n_training_samples, epochs=epochs, validation_data=(validation_inputs, validation_outputs), verbose=1, callbacks=[callback_early_stopping])
    print("Training time: %s seconds" % (time.time() - start_time))
    
   
    MSE = model.evaluate(testing_inputs, testing_outputs)
    print("MSE over test dataset: " + str(MSE[0]))

    
    y_train_pred = model.predict(training_inputs)
    y_val_pred = model.predict(validation_inputs)    
    y_test_pred = model.predict(testing_inputs)

    time_series_dataset.plot_generated_data(y_train_pred, y_val_pred, y_test_pred, outputs_all, plt_title="Predictions MLP for time series dataset")



 