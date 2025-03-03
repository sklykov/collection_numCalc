# -*- coding: utf-8 -*-
"""
Build and test non-linear multiparameters input-output model.

@author: sklykov
@license: The Unlicense

"""
# %% Imports
import numpy as np
import random
import time

scikit_available = False
try:
    import sklearn
    if sklearn is not None:
        scikit_available = True
except ModuleNotFoundError:
    pass


# %% Model definition - e.g. 4 input parameters interconnected and resulted in 4 output, scaled responses
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
                        calc_value = (a[i]-1.0)*el*el + a[i]*el + 0.01*random.randint(1, 6)*el
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
                        y[i] = interplay_coefficient*(1.0 - percentage_diff[i])*(a[i]*el + 0.01*random.randint(2, 6)*el)
                    else:
                        calc_value = (1.0 - percentage_diff[i])*((a[i]-1.0)*el*el + a[i]*el + 0.01*random.randint(1, 5)*el)
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
                        y[i] = interplay_coefficient*(0.8 - percentage_diff[i])*(a[i]*el + 0.01*random.randint(2, 7)*el)
                    else:
                        calc_value = (0.8 - percentage_diff[i])*((a[i]-1.0)*el*el + a[i]*el + 0.01*random.randint(1, 7)*el)
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
                    y[i] = (0.6 - percentage_diff[i])*(a[i]*el + 0.01*random.randint(2, 6)*el)
                else:
                    calc_value = (0.6 - percentage_diff[i])*((a[i]-1.0)*el*el + a[i]*el + 0.01*random.randint(1, 7)*el)
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


# %% Testing Model for input - output (Training data)
if __name__ == "__main__":
    x_input = []; y_data = []  # input and output data for testing ML regression
    # standard scanning scheme for getting data points (measure individual parameters)
    # x11 = [3.0, 0.0, 0.0, 0.0]; x12 = [1.5, 0.0, 0.0, 0.0]; x13 = [1.0, 0.0, 0.0, 0.0]; x14 = [0.5, 0.0, 0.0, 0.0]; x15 = [0.25, 0.0, 0.0, 0.0]
    # x21 = [0.0, 3.0, 0.0, 0.0]; x22 = [0.0, 1.5, 0.0, 0.0]; x23 = [0.0, 1.0, 0.0, 0.0]; x24 = [0.0, 0.5, 0.0, 0.0]; x25 = [0.0, 0.25, 0.0, 0.0]
    # x31 = [0.0, 0.0, 3.0, 0.0]; x32 = [0.0, 0.0, 1.5, 0.0]; x33 = [0.0, 0.0, 1.0, 0.0]; x34 = [0.0, 0.0, 0.5, 0.0]; x35 = [0.0, 0.0, 0.25, 0.0]
    # x41 = [0.0, 0.0, 0.0, 3.0]; x42 = [0.0, 0.0, 0.0, 1.5]; x43 = [0.0, 0.0, 0.0, 1.0]; x44 = [0.0, 0.0, 0.0, 0.5]; x45 = [0.0, 0.0, 0.0, 0.25]
    # x16 = [0.1, 0.0, 0.0, 0.0]; x26 = [0.0, 0.1, 0.0, 0.0]; x36 = [0.0, 0.0, 0.1, 0.0]; x46 = [0.0, 0.0, 0.0, 0.1]
    # x_input.append(x11); x_input.append(x12); x_input.append(x13); x_input.append(x14); x_input.append(x15); x_input.append(x16)
    # x_input.append(x21); x_input.append(x22); x_input.append(x23); x_input.append(x24); x_input.append(x25); x_input.append(x26)
    # x_input.append(x31); x_input.append(x32); x_input.append(x33); x_input.append(x34); x_input.append(x35); x_input.append(x36)
    # x_input.append(x41); x_input.append(x42); x_input.append(x43); x_input.append(x44); x_input.append(x45); x_input.append(x46)
    # Schematic above implemented in a loop below
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

    x_input = np.round(np.asarray(x_input), 3)

    # Getting output - Y
    for x_row in x_input:
        y = nonlinear_multiparam_model(x_row)
        y_data.append(y)
    y_data = np.asarray(y_data)

    # ML testing
    if scikit_available:
        # Testing kNN algorithm with different k of neighbours
        from sklearn import neighbors
        # from sklearn.metrics import r2_score
        from sklearn.metrics import root_mean_squared_error
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.multioutput import MultiOutputRegressor

        # kNN regression
        n_neighbors = 3
        knn3 = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        knn_model3 = knn3.fit(x_input, y_data)
        n_neighbors = 5
        knn5 = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        knn_model5 = knn5.fit(x_input, y_data)
        n_neighbors = 7
        knn7 = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        knn_model7 = knn7.fit(x_input, y_data)

        # Random forest (overall, more precise and stable than Nearest Neighbours)
        t1 = time.perf_counter(); n_estimators_rf = int(round(2.0*len(x_input), 0))  # increasing not equal to improving the accuracy
        default_rand_forest = RandomForestRegressor(n_estimators=n_estimators_rf)
        def_rand_forest_model = default_rand_forest.fit(x_input, y_data)
        print(f"Random Forest with {n_estimators_rf} estimators took ms for fitting:", int(round(1000.0*(time.perf_counter() - t1), 0)))

        # Gradient Boosting - more accurate than Random forest
        t1 = time.perf_counter(); n_estimators_gb = int(round(2.5*len(x_input), 0))
        # default_grad_boost = GradientBoostingRegressor(n_estimators=n_estimators_gb)  # doesn't support directly multi-dimension regression
        # def_grad_boost_model = default_grad_boost.fit(x_input, y_data)
        # MultiOutputRegressor wrapper supports multi dimension data
        multi_grad_boost = MultiOutputRegressor(GradientBoostingRegressor(learning_rate=0.04,
                                                                          n_estimators=n_estimators_gb))
        m_grad_boost_model = multi_grad_boost.fit(x_input, y_data)
        print(f"Multi Gradient Boost with {n_estimators_gb} estimators took ms for fitting:",
              int(round(1000.0*(time.perf_counter() - t1), 0)))

        # Making test data
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

        # Exact test data returned by the function (ground truth data)
        y_test = []
        for x_row in x_test:
            y = nonlinear_multiparam_model(x_row)
            y_test.append(y)
        y_test = np.round(np.asarray(y_test), 3)

        # Fitting test data and compare with the provided by the function
        y_fit3 = np.round(knn_model3.predict(x_test), 3)
        y_fit5 = np.round(knn_model5.predict(x_test), 3)
        y_fit7 = np.round(knn_model7.predict(x_test), 3)
        y_fit_def_rand_for = np.round(def_rand_forest_model.predict(x_test), 3)
        diff_fit_test_rand_for = np.round(np.abs(y_test - y_fit_def_rand_for), 3)
        y_fit_grad_boost = np.round(m_grad_boost_model.predict(x_test), 3)
        min_y_fit_grad_boost = np.min(y_fit_grad_boost)
        if min_y_fit_grad_boost < 0.0:
            y_fit_grad_boost += np.abs(min_y_fit_grad_boost)
        diff_fit_test_grad_boost = np.round(np.abs(y_test - y_fit_grad_boost), 3)

        # Estimation of model accuracy (based on R2 score function - not needed explicitly for KNeighborsRegressor)
        # Hint on R2 score values: 1.0 → Perfect fit, 0.9+ → Excellent fit, 0.5–0.9 → Moderate fit, < 0.5 → Poor fit
        # r2_score_knn3 = round(r2_score(y_test, y_fit3), 3)
        # r2_score_knn5 = round(r2_score(y_test, y_fit5), 3)
        # r2_score_knn7 = round(r2_score(y_test, y_fit7), 3)
        # Use native method for R2 score from KNeighborsRegressor
        r2_score_knn3 = round(knn3.score(x_test, y_test), 3)
        rmse_knn3 = round(root_mean_squared_error(y_test, y_fit3), 3)
        r2_score_knn5 = round(knn5.score(x_test, y_test), 3)
        rmse_knn5 = round(root_mean_squared_error(y_test, y_fit5), 3)
        r2_score_knn7 = round(knn7.score(x_test, y_test), 3)
        r2_score_def_rand_for = round(def_rand_forest_model.score(x_test, y_test), 3)
        rmse_def_rand_for = round(root_mean_squared_error(y_test, y_fit_def_rand_for), 3)
        r2_score_grad_boost = round(m_grad_boost_model.score(x_test, y_test), 3)
        rmse_grad_boost = round(root_mean_squared_error(y_test, y_fit_grad_boost), 3)
        print("kNN3 R2 score:", r2_score_knn3, " | RMSE:", rmse_knn3, "\n"
              + "kNN5 R2 score:", r2_score_knn5, " | RMSE:", rmse_knn5, "\n"
              + "kNN7 R2 score:", r2_score_knn7, " | ")
        print("R2 RandForest:", r2_score_def_rand_for, " | RMSE:", rmse_def_rand_for)
        # Even R2 score is good, difference between fit and calculated data is relatively big
        print("R2 GradiBoost:", r2_score_grad_boost, " | RMSE:", rmse_grad_boost,
              "\n")
