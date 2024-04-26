# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import statsmodels.formula.api as smf


from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    recall_score,
    precision_score,
    auc,
    roc_curve,
    log_loss
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
import random
import warnings

import pandas as pd
from statsmodels.formula.api import ols

def calculate_aic(model):
    # Assuming direct access to the AIC attribute is sufficient
    return model.aic

def AIC_forward_selection(data, dependent_var, predictors):
    """
    Perform AIC-based forward selection to identify a model with the lowest AIC.
    
    Parameters:
    - data: pandas DataFrame containing the dataset.
    - dependent_var: The name of the dependent variable (as a string).
    - predictors: A list of names (strings) of potential predictor variables.
    
    Returns:
    - A tuple of (selected_predictors, best_aic) indicating the model configuration
      with the lowest AIC and its AIC value.
    """
    selected_predictors = []
    current_aic = float('inf')
    
    while True:
        aic_with_candidates = []
        for predictor in predictors:
            # Try adding each remaining predictor to the selected predictors and calculate AIC
            formula = f"{dependent_var} ~ {' + '.join(selected_predictors + [predictor])}"
            model = ols(formula, data=data).fit()
            aic_with_candidates.append((model.aic, predictor))
        
        # Find the predictor that, when added, results in the lowest AIC
        aic_with_candidates.sort()
        best_new_aic, best_new_predictor = aic_with_candidates[0]
        
        # If adding the new variable lowers the AIC, update the model and continue
        if best_new_aic < current_aic:
            predictors.remove(best_new_predictor)
            selected_predictors.append(best_new_predictor)
            current_aic = best_new_aic
            print(f"Added {best_new_predictor}, AIC: {best_new_aic}")
        else:
            # If no improvement, stop the loop
            break
    
    return selected_predictors, current_aic


def forward_selection_pvalues(data, dependent_var, predictors):
    selected_predictors = []  # Start with no predictors
    remaining_predictors = predictors[:]
    pvalue_threshold = 0.05  # Define the significance level for inclusion
    
    while remaining_predictors:
        best_pvalue = 1  # Initialize with a value greater than any p-value
        best_predictor = None  # To keep track of the best predictor found in this iteration
        
        for predictor in remaining_predictors:
            if selected_predictors:
                formula = f"{dependent_var} ~ {' + '.join(selected_predictors)} + {predictor}"
            else:
                formula = f"{dependent_var} ~ {predictor}"
            model = smf.ols(formula, data=data).fit()
            # Check if predictor is in the model and then its p-value
            if predictor in model.params.index:
                pvalue = model.pvalues[predictor]
                if pvalue < best_pvalue:
                    best_pvalue = pvalue
                    best_predictor = predictor
        
        if best_predictor and best_pvalue < pvalue_threshold:
            print(f"Adding {best_predictor} with p-value {best_pvalue}")
            remaining_predictors.remove(best_predictor)
            selected_predictors.append(best_predictor)
        else:
            break  # Stop if no predictors improve the model significantly
    
    # Handle the case where no predictors have been selected
    if selected_predictors:
        final_formula = f"{dependent_var} ~ {' + '.join(selected_predictors)}"
    else:
        final_formula = f"{dependent_var} ~ 1"  # Fallback to a model with only an intercept
    
    final_model = smf.ols(final_formula, data=data).fit()
    return final_model



def evaluate(preds, actual):
    """
    Evaluate the performance of binary classification predictions.

    This function computes metrics including F1 score, accuracy, recall, and constructs a confusion matrix
    to provide detailed insight into the true positive, true negative, false positive, and false negative counts.

    Parameters:
    - preds (array-like): Predicted labels, must be binary (0 or 1).
    - actual (array-like): Actual true labels, must be binary (0 or 1).

    Returns:
    - dict: A dictionary containing the following metrics:
        - 'sensitivity' (float): Sensitivity, also known as recall or true positive rate.
        - 'specificity' (float): Specificity, a measure of how well the model identifies true negatives.
        - 'accuracy' (float): Accuracy of the predictions.
    - pandas DataFrame: Confusion matrix with the following structure:
        - Columns: 'PredDown' (predictions of the negative class) and 'PredUp' (predictions of the positive class).
        - Index labels: 'ActDown' (actual negative class) and 'ActUp' (actual positive class).

    Note:
    - This function assumes binary classification, and both preds and actual parameters should be of the same length.
    """

    acc = accuracy_score(actual, preds)
    
    tn, fp, fn, tp = confusion_matrix(actual, preds).ravel()
    spec = tn / (tn+fp)
    sens = tp / (tp + fn)

    cm = pd.DataFrame(
        confusion_matrix(actual,preds),
        columns=['PredDown','PredUp'],
        index=['ActDown','ActUp']
    )
    return {'sensitivity': sens, 'specificity ':spec, 'accuracy':acc}, cm

def diagnostic_plots(model, cooksd_prop=False):
    """
    Generate diagnostic plots for a regression model to assess the validity of model assumptions.
    
    This function creates a 2x2 grid of plots including Residuals vs Fitted, Q-Q plot, 
    Scale-Location plot, and Cook's Distance plot. These plots help to diagnose various 
    aspects of a regression model, such as linearity, homoscedasticity, and influential observations.
    
    Parameters:
    - model: A fitted regression model object from statsmodels.
    - cooksd_prop: A boolean flag. If True, the Cook's Distance plot will use a dynamic threshold
                   of 4/n (where n is the number of observations). Otherwise, a fixed threshold
                   of 1 is used. Default is False.

    Returns:
    - None: The function creates and displays the plot grid but does not return any values.
    
    Example usage:
    >>> model = sm.OLS(y, X).fit()
    >>> diagnostic_plots(model, cooksd_prop=True)
    """
    # Create a 2 by 2 subplot grid
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # =============================================================================
    # RESIDUALS VS FITTED
    # =============================================================================
    sns.residplot(
        x=model.fittedvalues,
        y=model.resid,
        lowess=True,
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red', 'lw': 3, 'alpha': 0.8},
        ax=axs[0, 0]
    )
    axs[0, 0].set_title('Residuals vs Fitted')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')
    
    # =============================================================================
    # QQ PLOT
    # =============================================================================
    # Q-Q plot with standardized residuals
    QQ = sm.ProbPlot(model.get_influence().resid_studentized_internal)
    QQ.qqplot(line='45', alpha=0.5, lw=0.5, ax=axs[0, 1])
    axs[0, 1].set_title('Q-Q Residuals')
    axs[0, 1].set_xlabel('Theoretical Quantiles')
    axs[0, 1].set_ylabel('Standardized Residuals')
    
    # =============================================================================
    # SCALE-LOCATION PLOT
    # =============================================================================
    standardized_resid = model.get_influence().resid_studentized_internal
    axs[1, 0].scatter(model.fittedvalues, np.sqrt(np.abs(standardized_resid)), alpha=0.5)
    axs[1, 0].set_title('Scale-Location')
    axs[1, 0].set_ylabel('√|Standardized residuals|')
    axs[1, 0].set_xlabel('Fitted values')

    # =============================================================================
    # COOKS DISTANCE PLOT
    # =============================================================================
    influence = model.get_influence()
    (c, p) = influence.cooks_distance
    axs[1, 1].stem(np.arange(len(c)), c, markerfmt=",", use_line_collection=True)
    axs[1, 1].set_title("Cook's distance")
    axs[1, 1].set_xlabel('Obs. number')
    axs[1, 1].set_ylabel("Cook's distance")

    # Draw Cook's distance threshold line
    if cooksd_prop:
        cooks_d_threshold = 4 / len(model.fittedvalues)  # Calculate the threshold
        label = f"Cook's d = {cooks_d_threshold:.2g}"
        threshold = cooks_d_threshold
    else:
        label = "Cook's d = 1"
        threshold = 1
    
    axs[1, 1].axhline(y=threshold, linestyle='--', color='orange', linewidth=1)
    axs[1, 1].text(x=np.max(np.arange(len(c)))*0.85, y=threshold*1.1, s=label, color='orange', va='bottom', ha='center')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()

def calculate_vif(X, y):
    """
    Calculates Variance Inflation Factors (VIF) for each feature in a dataset.

    This function adds a constant term to the predictor matrix, computes VIF for each feature, and returns a DataFrame listing each feature alongside its VIF.

    Parameters:
    - X (DataFrame): DataFrame containing predictor variables.
    - y (Series): Series containing the target variable. Note: `y` is not used in the function and can be removed.

    Returns:
    - DataFrame: A DataFrame with two columns: 'VIF', containing the VIF values, and 'col', listing the corresponding feature names.
    """
    
    # Add a constant term (intercept) to the feature matrix
    X_with_const = sm.add_constant(X)
    
    vif_vals = []
    for col in X_with_const.columns:
        vif = variance_inflation_factor(
            X_with_const.values,
            X_with_const.columns.get_loc(col)
        )
        vif_vals.append(vif)

    # Create a DataFrame with variable names and VIF values
    vif_df = pd.DataFrame({'VIF': vif_vals, 'col' : X_with_const.columns})
    
    return vif_df

def remove_high_vif_features(X, y, vif_threshold=5.0):
    """
    Removes features from the dataset X that have a Variance Inflation Factor (VIF) above a specified threshold.

    Iteratively calculates VIF for all features and removes the feature with the highest VIF if it exceeds the given threshold. This process is repeated until all features have VIF values below the threshold.

    Parameters:
    - X (DataFrame): DataFrame containing predictor variables.
    - y (Series): Series containing the target variable. Note: `y` is not used in the function and can be removed.
    - vif_threshold (float): The threshold above which a feature will be removed for having a high VIF value. Default is 5.0.

    Returns:
    - tuple:
        - DataFrame: The modified DataFrame with features removed that had a VIF above the threshold.
        - list: List of the names of the columns (features) that were removed.

    Note:
    - The function suppresses RuntimeWarnings during its execution and re-enables them afterward.
    """
    # Shut off RuntimeWarnings
    warnings.simplefilter("ignore", category=RuntimeWarning)
    
    # Keep track of the columns removed
    removed_Xs = []
    
    while True:
        vifs = calculate_vif(X, y)
        
        # Filter out 'const' from vifs DataFrame
        vifs = vifs[vifs['col'] != 'const']
        
        max_vif = vifs['VIF'].max()

        if max_vif > vif_threshold:
            max_vif_feature = vifs['col'][vifs['VIF'].idxmax()]
            removed_Xs.append(max_vif_feature)
            X = X.drop(columns=[max_vif_feature])
        else:
            break
    
    # Turn RuntimeWarnings back on
    warnings.simplefilter("default", category=RuntimeWarning)
    return X, removed_Xs
def plot_sensitivity_specificity(actual, preds):
    """
    Plot sensitivity (True Positive Rate) and specificity (1 - False Positive Rate) 
    against decision thresholds for a given set of actual binary labels and predicted 
    probabilities. This function also computes and displays the Area Under the 
    Curve (AUC) and identifies the optimal threshold where the absolute difference 
    between sensitivity and specificity is minimized.

    Parameters:
    - actual (array-like): True binary labels for the dataset. Must be a 1D array of 
      0s and 1s where 1 represents the positive class.
    - preds (array-like): Predicted probabilities for the positive class. Must be a 
      1D array with values ranging from 0 to 1, corresponding to the confidence level 
      of each prediction being in the positive class.

    Returns:
    - This function does not return any values. It generates a plot displaying the 
      sensitivity and specificity across different decision thresholds, highlights 
      the optimal threshold, and shows the AUC value in the plot title.
    """
    # Compute ROC curve and ROC area for each class
    fpr, tpr, thresholds = roc_curve(actual, preds)
    roc_auc = auc(fpr, tpr)

    # Calculate specificity: Specificity = 1 - FPR
    specificity = 1 - fpr

    # Plot sensitivity (TPR) vs. cutoff
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, tpr, label='Sensitivity', lw=2)
    plt.xlabel('Cutoff')
    plt.ylabel('Sensitivity')
    plt.title(f'Maximized Cutoff\nAUC: {roc_auc:.2f}')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Create new axis for specificity
    ax2 = plt.gca().twinx()
    ax2.plot(thresholds, specificity, label='Specificity', color='red', lw=2)
    ax2.set_ylabel('Specificity', color='red')
    ax2.tick_params(axis='y', colors='red')
    plt.yticks(np.arange(0, 1.1, 0.2))

    # Find the optimal threshold
    optimal_idx = np.argmin(np.abs(tpr - specificity))
    optimal_threshold = thresholds[optimal_idx]
    plt.axhline(y=specificity[optimal_idx], color='grey', linestyle='--')
    plt.axvline(x=optimal_threshold, color='grey', linestyle='--')
    plt.text(optimal_threshold, 0, f'Optimal threshold={optimal_threshold:.2f}', color='black', ha='right')

    plt.show()
    
def aic_scorer(model, X, y):
    """
    Calculates the Akaike Information Criterion (AIC) score for a given regression model.
    
    Parameters:
    - model: Trained regression model
    - X: Feature matrix
    - y: Target variable
    
    Returns:
    - AIC score
    """
    n = len(y)
    k = X.shape[1] + 1  # Number of features + intercept
    y_pred = model.predict(X)
    residuals = y - y_pred
    rss = np.sum(residuals ** 2)
    aic = n * np.log(rss / n) + 2 * k
    return aic


from tqdm import tqdm  # Import tqdm

def select_model_by_aic(X, y, feature_sets):
    """
    Selects the model with the lowest AIC score given a list of feature sets.
    
    Parameters:
    - X: DataFrame of features
    - y: Series of target variable
    - feature_sets: List of lists, where each sublist contains feature names for a model configuration
    
    Returns:
    - A tuple containing the best AIC score and the best feature set.
    """
    best_aic = np.inf
    best_features = None

    # Wrap feature_sets with tqdm for a progress bar
    for features in tqdm(feature_sets, desc="Evaluating feature sets"):
        X_subset = X[features]
        model = LinearRegression()  # Assuming you're using linear regression
        model.fit(X_subset, y)
        
        # Calculate AIC score
        aic_score = aic_scorer(model, X_subset, y)
        
        if aic_score < best_aic:
            best_aic = aic_score
            best_features = features
    
    return best_aic, best_features




def calculate_cooks_distance(X, y, model):
    """
    Fits a provided model to the data (X, y) and calculates Cook's distance for each observation.
    
    Parameters:
    - X: A NumPy array or pandas DataFrame containing the predictor variables.
    - y: A NumPy array or pandas Series containing the response variable.
    - model: A scikit-learn compatible model that implements the fit and predict_proba methods.
    
    Returns:
    - cooks_d: A NumPy array containing Cook's distance for each observation.
    """
    # Ensure X is a NumPy array if it's a pandas DataFrame
    X_np = X.values if hasattr(X, 'values') else X

    # Fit the provided model to the data
    model.fit(X_np, y)

    # Check if the model supports predict_proba, use it if available
    if hasattr(model, 'predict_proba'):
        y_pred_prob = model.predict_proba(X_np)[:, 1]
    else:
        # For models that do not support predict_proba, use predict method
        y_pred = model.predict(X_np)
        y_pred_prob = y_pred  # Adjust this line based on how you want to handle models without predict_proba
    
    # Calculate the residuals
    residuals = y - y_pred_prob

    # Calculate the leverage
    leverage = np.diag(X_np @ np.linalg.pinv(X_np.T @ X_np) @ X_np.T)

    # Calculate Cook's distance
    cooks_d = residuals**2 / (np.sum(residuals**2) * leverage)

    return cooks_d


def logistic_regression_diagnostic_plots(X, y, model, cookd_cutoff=0.5):
    """
    Generate diagnostic plots for a logistic regression model to assess the validity of model assumptions and
    identify influential observations based on Cook's distance.
    
    Parameters:
    - X: Features matrix.
    - y: Target variable.
    - model: A fitted logistic regression model object.
    - cookd_cutoff: The cutoff value for Cook's D to highlight influential points.

    Returns:
    - None: The function creates and displays the plot grid but does not return any values.
    """
    # Set up the figure
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))  # Adjust for 3 plots

    # Calculate diagnostics
    probs = model.predict_proba(X)[:, 1]
    deviance_residuals = -2 * (y * np.log(probs) + (1 - y) * np.log(1 - probs))
    std_deviance_residuals = deviance_residuals / np.std(deviance_residuals)
    pearson_residuals = (y - probs) / np.sqrt(probs * (1 - probs))
    std_pearson_residuals = pearson_residuals / np.std(pearson_residuals)
    cooks_d = calculate_cooks_distance(X, y, model)

    # Plot Standardized Deviance Residuals
    sns.scatterplot(x=range(len(std_deviance_residuals)), y=std_deviance_residuals, ax=axs[0])
    axs[0].set_xlabel('Observation Number')
    axs[0].set_ylabel('Std. Deviance Residuals')
    axs[0].set_title('Standardized Deviance Residuals')

    # Plot Standardized Pearson Residuals
    sns.scatterplot(x=range(len(std_pearson_residuals)), y=std_pearson_residuals, ax=axs[1])
    axs[1].set_xlabel('Observation Number')
    axs[1].set_ylabel('Std. Pearson Residuals')
    axs[1].set_title('Standardized Pearson Residuals')

    # Plot Cook's Distance
    sns.scatterplot(x=range(len(cooks_d)), y=cooks_d, ax=axs[2])
    cutoff_line = axs[2].axhline(y=cookd_cutoff, linestyle='--', color='red', linewidth=2)
    axs[2].set_xlabel('Observation Number')
    axs[2].set_ylabel('Cook\'s Distance')
    axs[2].set_title('Cook\'s Distance')
    # Add label for the horizontal line
    axs[2].text(len(cooks_d) * 0.85, cookd_cutoff, f'Cutoff: {cookd_cutoff}', verticalalignment='bottom', color='red')

    # Adjust layout to prevent overlapping
    plt.tight_layout()
    plt.show()


    
def view_logistic_regression_coefficients(model, X):
    """
    View the estimated coefficients for a logistic regression model along with their corresponding feature names.

    Parameters:
    - model: A fitted logistic regression model object.
    - feature_names: List or array containing the names of the features used in the model.

    Returns:
    - None: The function prints the coefficients DataFrame and the intercept but does not return any value.
    """
    # Get the estimated coefficients for the logistic regression model
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    
    feature_names = X.columns
    
    # Create a DataFrame to view the coefficients with their corresponding feature names
    coefs_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coef})

    # Sort the features by the absolute value of their coefficient
    coefs_df = coefs_df.reindex(coefs_df.Coefficient.abs().sort_values(ascending=False).index)

    # Display the DataFrame
    print(coefs_df)

    # Print the intercept
    print(f'Intercept: {intercept}')
    

