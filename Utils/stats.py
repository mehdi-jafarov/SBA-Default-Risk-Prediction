import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2
from sklearn.metrics import confusion_matrix

class Stats:
    @staticmethod
    def type3_test(target_col, variables, data):
        """
        Perform likelihood ratio test (Type 3 test) for each variable in logistic regression.

        Parameters:
        - target_col (str): Name of the dependent variable column.
        - variables (list of str): List of independent variable names.
        - data (pd.DataFrame): Dataset containing the variables.

        Returns:
        - pd.DataFrame: DataFrame with variables, degrees of freedom, chi-square statistics, and p-values.
        """
        full_formula = f'{target_col} ~ '  + ' + '.join(variables)
        full_model = smf.logit(full_formula, data).fit(disp=0)
        type3_results = []

        for var in variables:
            reduced_vars = [v for v in variables if v != var]
            reduced_formula = f'{target_col} ~ ' + ' + '.join(reduced_vars)
            reduced_model = smf.logit(reduced_formula, data).fit(disp=0)

            lr_stat = 2 * (full_model.llf - reduced_model.llf)
            df_diff = full_model.df_model - reduced_model.df_model
            p_value = chi2.sf(lr_stat, df_diff)

            type3_results.append({
                'Variable': var,
                'DF': int(df_diff),
                'Chi-Square': round(lr_stat, 2),
                'Pr > ChiSq': round(p_value, 4)
            })

        return pd.DataFrame(type3_results)

    @staticmethod
    def get_misclass_rates(cutoffs, y_prob, y_true):
        """
        Compute misclassification rates for given probability cutoffs.

        Parameters:
        - cutoffs (list or array-like): List of cutoff thresholds.
        - y_prob (array-like): Predicted probabilities.
        - y_true (array-like): True binary labels.

        Returns:
        - list: Misclassification rates corresponding to each cutoff.
        """
        misclass_rates = []
        for cutoff in cutoffs:
            y_pred = (y_prob > cutoff).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            misclass_rate = 1 - np.trace(cm) / np.sum(cm)
            misclass_rates.append(misclass_rate)

        return misclass_rates

    @staticmethod
    def get_conf_matrix(y_true, y_prob, cutoff=0.5, positive_label=1,
                        row_labels=None, col_labels=None):
        """
        Generate confusion matrix with totals and print misclassification rate.

        Parameters:
        - y_true (array-like): True binary labels.
        - y_prob (array-like): Predicted probabilities.
        - cutoff (float): Probability cutoff to determine predicted classes.
        - positive_label (int): Label to treat as positive class.
        - row_labels (list or None): Custom row labels for the confusion matrix.
        - col_labels (list or None): Custom column labels for the confusion matrix.

        Returns:
        - pd.DataFrame: Confusion matrix with totals.
        """
        y_pred = (y_prob > cutoff).astype(int)

        TP = np.sum((y_pred == positive_label) & (y_true == positive_label))
        TN = np.sum((y_pred != positive_label) & (y_true != positive_label))
        FP = np.sum((y_pred == positive_label) & (y_true != positive_label))
        FN = np.sum((y_pred != positive_label) & (y_true == positive_label))

        row_labels = row_labels or [f'Predicted {positive_label}', f'Predicted {1 - positive_label}']
        col_labels = col_labels or [f'Actual {positive_label}', f'Actual {1 - positive_label}']

        conf_matrix = pd.DataFrame({
            col_labels[0]: [TP, FN],
            col_labels[1]: [FP, TN],
        }, index=row_labels)

        conf_matrix['Total'] = conf_matrix.sum(axis=1)

        total_row = pd.DataFrame(conf_matrix.sum()).T
        total_row.index = ['Total']
        conf_matrix = pd.concat([conf_matrix, total_row])

        conf_matrix.index.name = 'Classification'

        misclass_rate = round(Stats.get_misclass_rates([cutoff], y_prob, y_true)[0] * 100, 2)
        print(f'Misclassification rate at cutoff {cutoff}: {misclass_rate}%')

        return conf_matrix
