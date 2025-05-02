# -*- coding: utf-8 -*-
"""
Build and test performance of regression for non-linear multiparameters input-output model.

@author: sklykov
@license: The Unlicense

"""
# %% Imports
import numpy as np
import random
import time
from pathlib import Path
import os
import math

# Checking that required libraries are installed
scikit_available = False
try:
    import sklearn
    if sklearn is not None:
        scikit_available = True
except ModuleNotFoundError:
    pass

joblib_available = False
try:
    import joblib
    if joblib is not None:
        joblib_available = True
except ModuleNotFoundError:
    pass

xgboost_available = False
try:
    import xgboost as xgb  # py-xgboost-cpu library from conda-forge is for minimal, CPU-based implementation
    if xgb is not None:
        xgboost_available = True
except ModuleNotFoundError:
    pass

# %% Script parameters, flags
test_saving_model = False  # test saving of fitted model using joblib serilization methods
check_diff_models = False


# %% Model definition - e.g. 4 input parameters nonlinearly interconnected and resulted in 4 output, scaled responses
def nonlinear_multiparam_model(x: np.array) -> np.array:
    """
    Provide nonlinear function mapping 4 input non-negative float point values into 4 outputs.

    Parameters
    ----------
    x : np.array
        Expected 4 non-negative elements.

    Returns
    -------
    np.array
        4 non-negative values.

    """
    if len(x.shape) == 1 and len(x) == 4:
        y = np.zeros(shape=x.shape)  # initialize returning coefficients
        a0 = 1.11; a1 = 1.07; a2 = 1.05; a3 = 1.02  # non-linear coefficients
        percentage_diff = [0.27, 0.19, 0.08, 0.11]  # reduction of amplitude if several non-zero amplitudes provided (diff from 1.0)
        max_value_replicated = [2.37, 2.48, 1.89, 2.16]  # the max coefficients that can be as output generated
        a = [a0, a1, a2, a3]  # storing in a list scaling coefficients
        intercoefficients_dependency = [[1.0, 0.95, 0.87, 1.0], [0.94, 1.0, 0.99, 0.92], [1.0, 1.0, 1.0, 0.81],
                                        [1.0, 1.0, 0.88, 1.0]]
        # Single amplitude provided (checking)
        n_zero_el = 0; i_nonzero_el = 0; i_nonzero_els = []; i_zero_els = []
        # Checking number of almost zero and non-zero elements
        for i, el in enumerate(x):
            if el <= 0.01:
                n_zero_el += 1; i_zero_els.append(i)
            else:
                i_nonzero_el = i; i_nonzero_els.append(i)
        if n_zero_el == 3:
            # Single not zero element, construct non-linear response
            for i, el in enumerate(x):
                if i == i_nonzero_el:
                    if el < 0.5:
                        y[i] = a[i]*el + 0.01*random.randint(1, 7)*el
                    else:
                        calc_value = (a[i]-1.0)*el*el + a[i]*el + 0.01*random.randint(1, 7)*el
                        if calc_value < max_value_replicated[i]:
                            y[i] = calc_value
                        else:
                            y[i] = max_value_replicated[i]
                else:
                    y[i] = random.choice([0.0, 0.01, 0.02, 0.03])
        elif n_zero_el == 2:
            for i, el in enumerate(x):
                if i in i_zero_els:
                    y[i] = random.choice([0.0, 0.01, 0.02, 0.03])
                else:
                    nonzero_i = i_nonzero_els[:]  # copy stored non-zero indices
                    index_i = nonzero_i.index(i); del nonzero_i[index_i]; i_other = nonzero_i[0]  # define other non-zero coefficient index
                    interplay_coefficient = intercoefficients_dependency[i][i_other]  # how variables influence each other
                    if el < 0.5:
                        y[i] = interplay_coefficient*(1.0 - percentage_diff[i])*(a[i]*el + 0.01*random.randint(2, 8)*el)
                    else:
                        calc_value = (1.0 - percentage_diff[i])*((a[i]-1.0)*el*el + a[i]*el + 0.01*random.randint(1, 6)*el)
                        calc_value *= interplay_coefficient
                        if calc_value < max_value_replicated[i]:
                            y[i] = calc_value
                        else:
                            y[i] = max_value_replicated[i]
        elif n_zero_el == 1:
            for i, el in enumerate(x):
                if i in i_zero_els:
                    y[i] = random.choice([0.0, 0.01, 0.02, 0.03])
                else:
                    nonzero_i = i_nonzero_els[:]  # copy stored non-zero indices
                    index_i = nonzero_i.index(i); del nonzero_i[index_i]; i_other1 = nonzero_i[0]; i_other2 = nonzero_i[1]
                    interplay_coefficient = intercoefficients_dependency[i][i_other1] + intercoefficients_dependency[i][i_other2]
                    interplay_coefficient *= 0.5
                    if el < 0.5:
                        y[i] = interplay_coefficient*(0.8 - percentage_diff[i])*(a[i]*el + 0.01*random.randint(2, 9)*el)
                    else:
                        calc_value = (0.8 - percentage_diff[i])*((a[i]-1.0)*el*el + a[i]*el + 0.01*random.randint(1, 8)*el)
                        calc_value *= interplay_coefficient
                        if calc_value < max_value_replicated[i]:
                            y[i] = calc_value
                        else:
                            y[i] = max_value_replicated[i]
        elif n_zero_el == 0:
            for i, el in enumerate(x):
                nonzero_i = i_nonzero_els[:]  # copy stored non-zero indices
                index_i = nonzero_i.index(i); del nonzero_i[index_i]; i_other1 = nonzero_i[0]; i_other2 = nonzero_i[1]  # define other indices
                i_other3 = nonzero_i[2]
                interplay_coefficient = intercoefficients_dependency[i][i_other1] + intercoefficients_dependency[i][i_other2]
                interplay_coefficient += intercoefficients_dependency[i][i_other3]; interplay_coefficient *= 0.33
                if el < 0.5:
                    y[i] = (0.6 - percentage_diff[i])*(a[i]*el + 0.01*random.randint(2, 7)*el)
                else:
                    calc_value = (0.6 - percentage_diff[i])*((a[i]-1.0)*el*el + a[i]*el + 0.01*random.randint(1, 8)*el)
                    if calc_value < max_value_replicated[i]:
                        y[i] = calc_value
                    else:
                        y[i] = max_value_replicated[i]
        else:
            for i in range(len(x)):
                y[i] = random.choice([0.0, 0.01, 0.02, 0.03])
        return y
    else:
        print("This model only resolves 4 inputs to 4 outputs")
        return x


# %% Testing finding a target (desired outcome) in a loop function
def find_best_target(target: np.ndarray, regressor, stop_precision: float = 1E-3) -> np.ndarray:
    """
    Find which input should be applied on the nonlinear function for obtaining this as the output.

    Parameters
    ----------
    target : np.ndarray
        Desired set of inputs that should be outputted from the nonlinear function.
    regressor : sciki tlearn class Regressor class with predict() method
        Instance of Regressor class (from in memory or read from a drive by 'joblib').
    stop_precision : float, optional
        Mean absolute error between target / prediction - Epsilon up to which optimization should run. The default is 1E-3.

    Returns
    -------
    updated_target : np.ndarray
        Input target that should be sent to the nonlinear function.

    """
    initial_predict = np.round(regressor.predict(target), 3)
    diff = np.round(target - initial_predict, 3)  # difference between desired target and predicted one
    updated_target = target + diff  # udpated target - that should be sent to an input for getting target as the output
    mean_diff_targets = np.mean(np.abs(diff))  # mean absolute errors between target / predicted sets
    previous_step_found_target = updated_target.copy()
    if mean_diff_targets > stop_precision:
        for i in range(10):
            predicted_target = np.round(regressor.predict(updated_target), 3)  # update predicted output
            diff = np.round(target - predicted_target, 3)  # update diff between initial input (desired target) and predicted
            # below - calculation for checking that each loop step guides to decreasing of difference between target and prediction
            optimization_step_diff = np.round(mean_diff_targets - np.mean(np.abs(diff)), 3)
            # If optimization is negative on the current iteration, below try to find some fraction of diff to achieve positive optimization
            if optimization_step_diff < 0.0:
                optimization_found = False
                j = 1  # iteration for applying portion of diff
                while j < 21:
                    if j != 10:
                        iter_diff = (1.0 - 0.1*j)*diff
                        predicted_target = np.round(regressor.predict(updated_target + iter_diff), 3)
                        diff_iter = np.round(target - predicted_target, 3)
                        optimization_step_diff_iter = np.round(mean_diff_targets - np.mean(np.abs(diff_iter)), 3)
                        # print(f"Used self correction of diff #{j}, {iter_diff}, {optimization_step_diff_iter}")
                        if optimization_step_diff_iter > 0.0:
                            optimization_found = True
                            break  # return if the optimization guides to butter
                    j += 1
                if optimization_found:
                    diff = diff_iter  # update diff from found iteration
                    optimization_step_diff = optimization_step_diff_iter  # update each step iteration efficiency
            mean_diff_targets = np.round(np.mean(np.abs(diff)), 3)  # update mean diff. (Mean Absolute Errors)
            # print(f"Step #{i} has: mean of diff(target-predicted): {optimization_step_diff}, mean diff: {mean_diff_targets}")
            # print(f"Step #{i} has: Diff. target - predicted target:", diff)
            if mean_diff_targets <= stop_precision or optimization_step_diff <= 0.0:
                updated_target = previous_step_found_target.copy()
                # print("Assigned from previous step target:", updated_target)
                break  # return the defined updated_target on a previous step
            previous_step_found_target = updated_target.copy()  # store found on the previous step target if the new one worsen
            updated_target = np.round(updated_target + diff, 3)  # update target to reuse it in the outer for loop
            # print(f"Step #{i} found updated target:", updated_target)
    # print("Found target to be sent:", updated_target)
    return updated_target


# %% Fitting various models, tuning parameters, comparing scores for them
if __name__ == "__main__":
    x_input = []; y_data = []  # input and output data for testing ML regression (fitting)
    # %% Data / measurements  preparation
    # Standard scanning scheme for getting data points (measure individual parameters), appending collected data as rows
    for i in range(4):
        x_step = [0.0, 0.0, 0.0, 0.0]
        for j in range(6):
            x_step = [0.0, 0.0, 0.0, 0.0]
            if j == 0:
                x_step[i] = 3.0
            elif j == 1:
                x_step[i] = 1.5
            elif j == 2:
                x_step[i] = 1.0
            elif j == 3:
                x_step[i] = 0.5
            elif j == 4:
                x_step[i] = 0.25
            elif j == 5:
                x_step[i] = 0.1
            x_input.append(x_step)

    # Interconnected 2 parameters scanning
    use_scanning_guess = True  # build input data to scan better on all cases
    if use_scanning_guess:
        x4fit = [0.0, 0.0, 0.0, 0.0]; scan_ampl_init = 0.25
        for j in range(4):
            scan_ampl = scan_ampl_init*(j+1)
            for i in range(4):
                indices = [0, 1, 2, 3]; x4fit = [0.0, 0.0, 0.0, 0.0]
                x4fit[i] = scan_ampl; xscanning = x4fit[:]
                scanning_indices = indices[:]; del scanning_indices[i]
                for scan_i in scanning_indices:
                    xscanning[scan_i] = scan_ampl
                    if xscanning not in x_input:
                        x_input.append(xscanning)
                    xscanning = x4fit[:]
        # Interconnected 3 parameters scanning
        x4fit = [0.0, 0.0, 0.0, 0.0]; scan_ampl_init = 0.25
        for j in range(4):
            scan_ampl = scan_ampl_init*(j+1)
            for i in range(3):
                indices = [0, 1, 2, 3]; x4fit = [0.0, 0.0, 0.0, 0.0]
                x4fit[i] = scan_ampl; x4fit[i+1] = scan_ampl; xscanning = x4fit[:]
                scanning_indices = indices[:]; del scanning_indices[i:i+1]
                for scan_i in scanning_indices:
                    xscanning[scan_i] = scan_ampl
                    if xscanning not in x_input:
                        x_input.append(xscanning)
                    xscanning = x4fit[:]

    # Making random combinations of input parameters for fitting
    x1_max = 2.0; x2_max = 2.0; x3_max = 1.5; x4_max = 1.75
    n_overall = 120  # number of points to learn overall
    n_randomized_points = n_overall - len(x_input)
    for i in range(n_randomized_points):
        zero_ampls = random.choice([0, 1, 2])  # choice how many zero amplitudes will be used in combinations
        zeroing_coefficients = [1.0, 1.0, 1.0, 1.0]
        if zero_ampls == 1:
            zero_ampl = random.choice([0, 1, 2])  # select zero coefficient
            zeroing_coefficients[zero_ampl] = 0.0
        elif zero_ampls == 2:
            zero_ampl1 = random.choice([0, 1, 2, 3]); indices = [0, 1, 2, 3]; del indices[zero_ampl1]
            zero_ampl2 = random.choice(indices)  # select 2nd zero coefficient
            zeroing_coefficients[zero_ampl1] = 0.0; zeroing_coefficients[zero_ampl2] = 0.0
        # Compose random amplitudes in combination
        x1 = zeroing_coefficients[0]*random.random()*x1_max
        x2 = zeroing_coefficients[1]*random.random()*x2_max
        x3 = zeroing_coefficients[2]*random.random()*x3_max
        x4 = zeroing_coefficients[3]*random.random()*x4_max
        x_row = [x1, x2, x3, x4]; x_input.append(x_row)  # add to the data

    x_input = np.round(np.asarray(x_input), 3)  # input parameters on which Y data depending

    # Getting output - Y data
    for x_row in x_input:
        y = nonlinear_multiparam_model(x_row)
        y_data.append(y)
    y_data = np.asarray(y_data)

    # Making input test data
    x_test = []; x_test.append([0.0, 0.3, 0.0, 0.0])
    for i in range(len(x_input)*3):
        zero_ampls = random.choice([0, 1, 2, 3])  # choice how many zero amplitudes will be used in combinations
        zeroing_coefficients = [1.0, 1.0, 1.0, 1.0]
        if zero_ampls == 1:
            zero_ampl = random.choice([0, 1, 2])  # select zero coefficient
            zeroing_coefficients[zero_ampl] = 0.0
        elif zero_ampls == 2:
            zero_ampl1 = random.choice([0, 1, 2, 3]); indices = [0, 1, 2, 3]; del indices[zero_ampl1]
            zero_ampl2 = random.choice(indices)  # select 2nd zero coefficient
            zeroing_coefficients[zero_ampl1] = 0.0; zeroing_coefficients[zero_ampl2] = 0.0
        elif zero_ampls == 3:
            indices = [0, 1, 2, 3]
            zero_ampl1 = random.choice(indices); del indices[zero_ampl1]
            zero_ampl2 = random.choice(indices); del indices[indices.index(zero_ampl2)]  # select 2nd zero coefficient
            zero_ampl3 = random.choice(indices)
            zeroing_coefficients[zero_ampl1] = 0.0; zeroing_coefficients[zero_ampl2] = 0.0; zeroing_coefficients[zero_ampl3] = 0.0
        # Compose random amplitudes in combination
        x1 = zeroing_coefficients[0]*random.random()*x1_max
        x2 = zeroing_coefficients[1]*random.random()*x2_max
        x3 = zeroing_coefficients[2]*random.random()*x3_max
        x4 = zeroing_coefficients[3]*random.random()*x4_max
        x_row = [x1, x2, x3, x4]; x_test.append(x_row)  # add to the data
    x_test = np.asarray(x_test)  # generated test data

    # Exact test data returned by the function (ground truth data) for comparing with fitted one
    y_test = []
    for x_row in x_test:
        y = nonlinear_multiparam_model(x_row)
        y_test.append(y)
    y_test = np.round(np.asarray(y_test), 3)

    # %% ML Regression testing
    if scikit_available:
        from sklearn import neighbors  # kNN algorithm - fastest and simplest, but less accurate
        # from sklearn.metrics import r2_score  # all models implement this score as a method
        from sklearn.metrics import root_mean_squared_error
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor  # wraps up all GBR regressors for fitting 2D matrix as an input
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # used for tuning parameters used by GBR models

        # kNN regression
        n_neighbors = 3; knn3 = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        knn_model3 = knn3.fit(x_input, y_data)
        n_neighbors = 5; knn5 = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        knn_model5 = knn5.fit(x_input, y_data)
        n_neighbors = 7; knn7 = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        knn_model7 = knn7.fit(x_input, y_data)

        # Random forest (overall, more precise and stable than Nearest Neighbours but less precise than GBR models)
        t1 = time.perf_counter(); n_estimators_rf = int(round(2.0*len(x_input), 0))  # increasing not equal to improving the accuracy
        default_rand_forest = RandomForestRegressor(n_estimators=n_estimators_rf)
        def_rand_forest_model = default_rand_forest.fit(x_input, y_data)
        print("\nEvaluation of individual fitting performance")
        print(f"Random Forest with {n_estimators_rf} estimators took ms for fitting:", int(round(1000.0*(time.perf_counter() - t1), 0)))

        # Gradient Boosting - more accurate than Random forest
        t1 = time.perf_counter(); n_estimators_gb = int(round(2.0*len(x_input), 0))
        # MultiOutputRegressor supports multidimension input, GradientBoostingRegressor doesn't support directly multi-dimension regression
        learn_rate_gbr = 0.04  # parameter for both GradientBoostingRegressor below and for XGBRegressor
        multi_grad_boost = MultiOutputRegressor(GradientBoostingRegressor(learning_rate=learn_rate_gbr, n_estimators=n_estimators_gb))
        m_grad_boost_model = multi_grad_boost.fit(x_input, y_data)
        print(f"GBR with {n_estimators_gb} estimators took ms for fitting:", int(round(1000.0*(time.perf_counter() - t1), 0)))

        # Testing XGBoost regression. It is much faster for fitting and tuning than GBR from scikit-learn, but less precise
        multi_xgb_regressor = None
        if xgboost_available:
            t1 = time.perf_counter()
            multi_xgb_regressor = MultiOutputRegressor(xgb.XGBRegressor(learning_rate=learn_rate_gbr, n_estimators=n_estimators_gb))
            multi_xgb_regressor.fit(x_input, y_data)
            print(f"XGBRegressor with {n_estimators_gb} estimators took ms for fitting:", int(round(1000.0*(time.perf_counter() - t1), 0)))

        # Testing the grid search for optimal hyperparameters tuning for GBR model
        # Building up number of estimators based on the input data length
        n_estimators_gb_range = [int(round(0.48*(i+1)*len(x_input), 0)) for i in range(8)]  # makes it not equal stepping - % of length used
        # prefixes 'estimator_' before parameters naming are used because of wrapper MultiOutputRegressor. For single - use without prefix
        # Note: search duration grows as combination of parameters below
        param_grid = {
            'estimator__n_estimators': n_estimators_gb_range,
            'estimator__learning_rate': [0.025, 0.05, 0.075, 0.1],
            'estimator__max_depth': [3, 5, 7]
            # 'estimator__subsample': [0.85, 0.9, 1.0]  # fraction of samples to be randomly selected for fitting each individual tree
            # subsample parameter switched off, because the input data maybe be not enough in the case of limited measurements
        }
        available_threads = os.cpu_count(); assigned_threads = 1
        if available_threads > 2:
            assigned_threads = available_threads - 1  # retain responsivness by not assigning 1 thread (it's sufficiently enough)
        # below: MSE - standard score, n_jobs=-1 - use all available threads
        print("\n**** GridSearchCV started for tuning hyperparameters (GBR)... ****"); t1 = time.perf_counter()
        # Note that final time required for search below is defined as N parameters combination x N cv (controlled by cv parameter below)
        n_params_combinations = [len(value) for value in param_grid.values()]
        cv = 5  # number of cross-validation folds - splits data into only these folds, cv-1 used for training, 1 - for validation
        print("Number of required fits for performing Grid Search: ", cv*math.prod(n_params_combinations))
        grid_gbr = GridSearchCV(estimator=MultiOutputRegressor(GradientBoostingRegressor()), param_grid=param_grid,
                                scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)  # use all threads, let OS to distribute tasks
        grid_gbr.fit(x_input, y_data)
        print("GridSearch (GBR) took min. for get done:", round((time.perf_counter() - t1)/60.0, 2))
        # Important note  - retrain the GBR on the found set of best parameters, remove prefix 'estimator__' from keys
        params_cleaned = {key.replace("estimator__", ""): value for key, value in grid_gbr.best_params_.items()}
        t1 = time.perf_counter(); gbr_best_params = MultiOutputRegressor(GradientBoostingRegressor(**params_cleaned))
        gbr_best_params.fit(x_input, y_data)
        print(f"GBR with found params on Grid {params_cleaned} took ms for fitting:", int(round(1000.0*(time.perf_counter() - t1), 0)), "\n")

        # Testing grid search with same parameters but for XGB regressor (always much faster than using native GradientBoostingRegressor)
        if xgboost_available:
            print("**** GridSearchCV started for tuning hyperparameters (XGB)... ****"); t1 = time.perf_counter()
            grid_xgb = GridSearchCV(estimator=MultiOutputRegressor(xgb.XGBRegressor()), param_grid=param_grid,
                                    scoring='neg_mean_squared_error', n_jobs=assigned_threads, cv=cv)
            grid_xgb.fit(x_input, y_data)
            print("GridSearch (XGB) took min. for get done:", round((time.perf_counter() - t1)/60.0, 2))
            params_cleaned = {key.replace("estimator__", ""): value for key, value in grid_xgb.best_params_.items()}
            t1 = time.perf_counter()
            xgb_best_params = MultiOutputRegressor(xgb.XGBRegressor(**params_cleaned))
            xgb_best_params.fit(x_input, y_data)
            print(f"XGB with found params {params_cleaned} took ms for fitting:", int(round(1000.0*(time.perf_counter() - t1), 0)), "\n")

        # Testing performance and accuracy found best parameters on a grid search for HistGradientBoostingRegressor
        print("**** GridSearchCV started for tuning hyperparameters (Hist GBR)... ****"); t1 = time.perf_counter()
        del param_grid['estimator__n_estimators']; param_grid['estimator__max_iter'] = n_estimators_gb_range  # exchange name of parameter
        grid_histgbr = GridSearchCV(estimator=MultiOutputRegressor(HistGradientBoostingRegressor()), param_grid=param_grid,
                                    scoring='neg_mean_squared_error', n_jobs=assigned_threads, cv=cv)
        grid_histgbr.fit(x_input, y_data)
        print("GridSearch (Hist GBR) took min. for get done:", round((time.perf_counter() - t1)/60.0, 2))
        params_cleaned = {key.replace("estimator__", ""): value for key, value in grid_histgbr.best_params_.items()}
        t1 = time.perf_counter(); histgbr_best_params = MultiOutputRegressor(HistGradientBoostingRegressor(**params_cleaned))
        histgbr_best_params.fit(x_input, y_data)
        print(f"Hist GBR with found params {params_cleaned} took ms for fitting:", int(round(1000.0*(time.perf_counter() - t1), 0)), "\n")

        # Comparing results of deterministic GridSearch with RandomSearch that is overall faster and allows wider range of parameters
        n_estimators_min = int(round(0.48*len(x_input), 0)); n_estimators_max = 9*n_estimators_min + 2
        param_grid = {
            'estimator__n_estimators': np.arange(start=n_estimators_min, stop=n_estimators_max, step=n_estimators_min),
            'estimator__learning_rate': np.round(np.linspace(0.01, 0.1, 9), 3),
            'estimator__max_depth': np.arange(3, 8, 2)
        }
        n_iterations = 30  # defines how many fits will be done (restricting number of computations)
        print("**** RandomizedSearchCV started for tuning hyperparameters (GBR)... ****"); t1 = time.perf_counter()
        rand_gbr = RandomizedSearchCV(estimator=MultiOutputRegressor(GradientBoostingRegressor()), param_distributions=param_grid,
                                      scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)  # use all threads, let OS to distribute tasks
        rand_gbr.fit(x_input, y_data)
        print("RandomizedSearch (GBR) took min. for get done:", round((time.perf_counter() - t1)/60.0, 2))
        # Important note  - retrain the GBR on the found set of best parameters, remove prefix 'estimator__' from keys
        params_cleaned = {key.replace("estimator__", ""): int(value) if 'learning_rate' not in key else round(float(value), 3)
                          for key, value in rand_gbr.best_params_.items()}
        t1 = time.perf_counter(); gbr_rand_params = MultiOutputRegressor(GradientBoostingRegressor(**params_cleaned))
        gbr_rand_params.fit(x_input, y_data)
        print(f"GBR with found params on Rand. {params_cleaned} took ms for fitting:", int(round(1000.0*(time.perf_counter() - t1), 0)), "\n")

        if xgboost_available:
            print("**** RandomizedSearchCV started for tuning hyperparameters (XGB)... ****"); t1 = time.perf_counter()
            rand_xgb = RandomizedSearchCV(estimator=MultiOutputRegressor(xgb.XGBRegressor()), param_distributions=param_grid,
                                          scoring='neg_mean_squared_error', n_jobs=-1, cv=cv, n_iter=n_iterations)
            rand_xgb.fit(x_input, y_data)
            print("RandomizedSearch (XGB) took min. for get done:", round((time.perf_counter() - t1)/60.0, 2))
            params_cleaned = {key.replace("estimator__", ""): int(value) if 'learning_rate' not in key else round(float(value), 3)
                              for key, value in rand_xgb.best_params_.items()}
            t1 = time.perf_counter()
            xgb_rand_params = MultiOutputRegressor(xgb.XGBRegressor(**params_cleaned)); xgb_rand_params.fit(x_input, y_data)
            print(f"Rand XGB with found params {params_cleaned} took ms for fitting:", int(round(1000.0*(time.perf_counter() - t1), 0)), "\n")

        # Fitting test data and compare with the provided by the function (measurements)
        y_fit3 = np.round(knn_model3.predict(x_test), 3); y_fit5 = np.round(knn_model5.predict(x_test), 3)
        y_fit7 = np.round(knn_model7.predict(x_test), 3); y_fit_def_rand_for = np.round(def_rand_forest_model.predict(x_test), 3)
        y_fit_grad_boost = np.round(m_grad_boost_model.predict(x_test), 3); y_fit_grid_gbr = np.round(gbr_best_params.predict(x_test), 3)
        y_fit_hist_gbr = np.round(histgbr_best_params.predict(x_test), 3); y_fit_rand_gbr = np.round(gbr_rand_params.predict(x_test), 3)
        if multi_xgb_regressor is not None:
            y_fit_xgb = np.round(multi_xgb_regressor.predict(x_test), 3); y_fit_xgb_best = np.round(xgb_best_params.predict(x_test), 3)
            y_fit_xgb_rand = np.round(xgb_rand_params.predict(x_test), 3)
        if check_diff_models:
            diff_fit_test_rand_for = np.round(np.abs(y_test - y_fit_def_rand_for), 3)
            diff_fit_test_grad_boost = np.round(np.abs(y_test - y_fit_grad_boost), 3)

        # Estimation of model accuracy (based on R2 score function - not needed explicitly for KNeighborsRegressor)
        # Hint on R2 score values: 1.0 → Perfect fit, 0.9+ → Excellent fit, 0.5–0.9 → Moderate fit, < 0.5 → Poor fit
        r2_score_knn3 = round(knn3.score(x_test, y_test), 3); rmse_knn3 = round(root_mean_squared_error(y_test, y_fit3), 3)
        r2_score_knn5 = round(knn5.score(x_test, y_test), 3); rmse_knn5 = round(root_mean_squared_error(y_test, y_fit5), 3)
        r2_score_knn7 = round(knn7.score(x_test, y_test), 3); r2_score_def_rand_for = round(def_rand_forest_model.score(x_test, y_test), 3)
        rmse_def_rand_for = round(root_mean_squared_error(y_test, y_fit_def_rand_for), 3)
        r2_score_grad_boost = round(m_grad_boost_model.score(x_test, y_test), 3)
        rmse_grad_boost = round(root_mean_squared_error(y_test, y_fit_grad_boost), 3)
        r2_score_grid_gbr = round(gbr_best_params.score(x_test, y_test), 3)
        rmse_grid_gbr = round(root_mean_squared_error(y_test, y_fit_grid_gbr), 3)
        r2_score_hist_gbr = round(histgbr_best_params.score(x_test, y_test), 3)
        rmse_hist_gbr = round(root_mean_squared_error(y_test, y_fit_hist_gbr), 3)
        r2_score_rand_gbr = round(gbr_rand_params.score(x_test, y_test), 3)
        rmse_rand_gbr = round(root_mean_squared_error(y_test, y_fit_rand_gbr), 3)
        if multi_xgb_regressor is not None:
            r2_score_xgb = round(multi_xgb_regressor.score(x_test, y_test), 3)
            rmse_xgb = round(root_mean_squared_error(y_test, y_fit_xgb), 3)
            r2_score_xgb_best = round(xgb_best_params.score(x_test, y_test), 3)
            rmse_xgb_best = round(root_mean_squared_error(y_test, y_fit_xgb_best), 3)
            r2_score_xgb_rand = round(xgb_rand_params.score(x_test, y_test), 3)
            rmse_xgb_rand = round(root_mean_squared_error(y_test, y_fit_xgb_rand), 3)
        print("***** Evaluation of model precision *****")
        print("kNN3 R2 score:", r2_score_knn3, " | RMSE:", rmse_knn3, "\n"
              + "kNN5 R2 score:", r2_score_knn5, " | RMSE:", rmse_knn5, "\n"
              + "kNN7 R2 score:", r2_score_knn7, " | ")
        print("R2 RandForest:", r2_score_def_rand_for, " | RMSE:", rmse_def_rand_for)
        # Even R2 score is good, difference between fit and calculated data is relatively big
        print("R2 Def. GBR:  ", r2_score_grad_boost, " | RMSE:", rmse_grad_boost)
        print("R2 Grid GBR:  ", r2_score_grid_gbr, " | RMSE:", rmse_grid_gbr)
        print("R2 Hist GBR:  ", r2_score_hist_gbr, " | RMSE:", rmse_hist_gbr)
        print("R2 Rand GBR:  ", r2_score_rand_gbr, " | RMSE:", rmse_rand_gbr)
        if multi_xgb_regressor is not None:
            print("R2 XGB Regr.: ", r2_score_xgb, " | RMSE:", rmse_xgb)
            print("R2 XGB Grid : ", r2_score_xgb_best, " | RMSE:", rmse_xgb_best)
            print("R2 XGB Rand : ", r2_score_xgb_rand, " | RMSE:", rmse_xgb_rand)
        print()

        # Making fitted model persistent for reusing it, compared fitting with the model saved in memory
        if joblib_available and test_saving_model:
            root_script_path = Path(__name__).parent
            gradiboostregr_model_name = "gradiboostmodel.joblib"
            model_path = root_script_path.joinpath(gradiboostregr_model_name)
            joblib.dump(value=m_grad_boost_model, filename=model_path, compress=3)
            if model_path.exists() and model_path.is_file():
                m_grad_boost_model_read = joblib.load(model_path)
                y_fit_grad_boost_read = np.round(m_grad_boost_model_read.predict(x_test), 3)
                diff_read_memory_grad_boost = np.round(np.abs(y_fit_grad_boost - y_fit_grad_boost_read), 3)  # 0.0 difference
        elif test_saving_model and not joblib_available:
            print("Install 'joblib' library for testing saving of a fitted model")
        else:
            del joblib_available, test_saving_model

        # Developing and testing finding of best fit for a target
        x_target = np.asarray([0.04, 0.39, 0.0, 0.72]).reshape((1, -1))  # reshape required because the model fitted with 2D array
        x_target_found = find_best_target(x_target, m_grad_boost_model)
        x_target_predicted = np.round(m_grad_boost_model.predict(x_target_found), 3)
        x_target_measured = np.round(nonlinear_multiparam_model(x_target_found[0]), 3)
        diff_target_predicted = np.round(x_target - x_target_predicted, 3)
        mean_diff_target_predicted = np.round(np.mean(np.abs(diff_target_predicted)), 3)
        diff_target_measured = np.round(x_target - x_target_measured, 3)
        mean_diff_target_measured = np.round(np.mean(np.abs(diff_target_measured)), 3)
        print("Difference desired / predicted target:", diff_target_predicted, " | Mean abs. diff.:", mean_diff_target_predicted)
        print("Difference desired / measured after applying optimized target:", diff_target_measured,
              " | Mean abs. diff.:", mean_diff_target_measured, "\n")

        # Testing the same logic but with best found on a grid search GBR model
        print("Using GBR model with best found parameters on a grid search")
        x_target_found = find_best_target(x_target, gbr_best_params)
        x_target_predicted = np.round(gbr_best_params.predict(x_target_found), 3)
        x_target_measured = np.round(nonlinear_multiparam_model(x_target_found[0]), 3)
        diff_target_predicted = np.round(x_target - x_target_predicted, 3)
        mean_diff_target_predicted = np.round(np.mean(np.abs(diff_target_predicted)), 3)
        diff_target_measured = np.round(x_target - x_target_measured, 3)
        mean_diff_target_measured = np.round(np.mean(np.abs(diff_target_measured)), 3)
        print("Difference desired / predicted target:", diff_target_predicted, " | Mean abs. diff.:", mean_diff_target_predicted)
        print("Difference desired / measured after applying optimized target:", diff_target_measured,
              " | Mean abs. diff.:", mean_diff_target_measured, "\n")

        # Clean up variables from excluding them from Variable Explorer in Spyder (for easier inspection)
        del i, j, scan_ampl, scan_ampl_init, scan_i, scanning_indices, t1, x1, x2, x3, x4, x1_max, x2_max, x3_max, x4_max
        del x4fit, x_row, x_step, xscanning, zero_ampl, zero_ampl1, zero_ampl2, zero_ampl3, zero_ampls, zeroing_coefficients
        del n_estimators_gb, n_estimators_rf, n_neighbors, n_overall, n_randomized_points, r2_score_knn3, r2_score_knn5
        del r2_score_def_rand_for, r2_score_knn7, rmse_knn3, rmse_knn5, rmse_grad_boost, rmse_def_rand_for, r2_score_grad_boost
        del xgboost_available, use_scanning_guess, scikit_available, n_estimators_max, n_estimators_min, n_iterations
        del rmse_grid_gbr, rmse_hist_gbr, rmse_rand_gbr, rmse_xgb, rmse_xgb_best, rmse_xgb_rand
