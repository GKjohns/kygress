import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def regress(
    X,
    y,
    df,
    keep_intercept=False,
    fdr_correction=None,
    significance_level=.05,
    formula=None):

    def format_cat_var(cat_var):
        '''hacky af'''

        if not cat_var.startswith('C('):
            return cat_var

        var_name = cat_var.split(')[')[0][2:]
        cat_name = cat_var.split(')[T.')[1][:-1]

        return f'{var_name}_is_{cat_name}'

    def is_stat_sig(p_vals, significance_level_, fdr_correction_):

        # if fdr_correction_ and fdr_correction_ in (True, 'fdr'):
        #     ranks = np.argsort(abs(p_vals)) + 1
        #     thresholds = significance_level_ * ranks / len(p_vals)
        if fdr_correction_:
            raise NotImplementedError('Need to implement fdr correction, use None!')
        else:
            thresholds = significance_level_

        return p_vals < thresholds

    def regress_(X_, y_, df_, keep_intercept_, fdr_correction_, significance_level_, formula_):
        '''regression on a single target'''

        formula_ = formula_ if formula_ else f'{y_} ~ {" + ".join(X_)}'
        model = smf.ols(formula_, data=df_.dropna(how='any')).fit()

        df_out = (
            pd.DataFrame({
                'target': y_,
                'coef': model.params,
                't_stat': model.tvalues,
                'p_val': model.pvalues,
                'ci_lower': model.conf_int()[0],
                'ci_upper': model.conf_int()[1]
            })
            .reset_index()
            .rename(columns={'index': 'param'})
            .replace({'param': {'Intercept': 'intercept'}})
            .assign(
                is_stat_sig=lambda d: is_stat_sig(d.p_val, significance_level_, fdr_correction_),
                param=lambda d: d.param.apply(format_cat_var)
            )
            .astype({'is_stat_sig': np.int8})
        )

        if not keep_intercept:
            df_out = df_out.query('param != "intercept"')

        return (
            df_out
            .sort_values(['param', 'target'])
            .set_index(['param', 'target'])
        )

    if isinstance(y, list):   # single target case
        return pd.concat([regress_(X, target, df, keep_intercept, fdr_correction, significance_level, formula) for target in y])
    return regress_(X, y, df, keep_intercept, fdr_correction, significance_level, formula)