import scipy.stats
from scipy.stats import norm

import pandas as pd
import numpy as np
from numpy.linalg import inv
import statsmodels.api as sm

def annualize_rets(r, periods_per_year=12):
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year=12):
    return r.std()*(periods_per_year**0.5)



def sharpe_ratio(r, riskfree_rate, periods_per_year):
    excess_ret = r.subtract(riskfree_rate, axis=0)
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year=12)
    ann_vol = annualize_vol(r, periods_per_year=12)
    return ann_ex_ret/ann_vol


def drawdown(return_series: pd.Series):
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })


def skewness(r):
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    demeaned_r = r - r.mean()
    # use the population standard deviation, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4


def var_gaussian(r, level=5, modified=False):
    # compute the Z score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
                 (z**2 - 1)*s/6 +
                 (z**3 - 3*z)*(k-3)/24 -
                 (2*z**3 - 5*z)*(s**2)/36
            )        
    return -(r.mean() + z*r.std(ddof=0))


def var_historic(r, level=5):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else: 
        raise TypeError("Expected r to be Series or DataFrame")

        
def cvar_historic(r, level=5):
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")




def summary_stats(r, riskfree_rate):
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = sharpe_ratio(r, riskfree_rate, periods_per_year=12)
    #ann_sr = sharpe_ratio(r, riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_var5 = r.aggregate(var_historic)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic VaR (5%)": hist_var5,
        "Historic CVaR (5%)": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })


def stats(r, riskfree_rate):
    ann_r = r.aggregate(annualize_rets, periods_per_year=12)
    ann_vol = r.aggregate(annualize_vol, periods_per_year=12)
    ann_sr = sharpe_ratio(r, riskfree_rate, periods_per_year=12)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })
#------------------------------- Risk Parity Unlevered -------------------------------------------


from scipy.optimize import minimize
def risk_contribution(w,cov):
    total_portfolio_var = portfolio_vol(w,cov)**2
    # Marginal contribution of each constituent
    marginal_contrib = cov@w
    risk_contrib = np.multiply(marginal_contrib,w.T)/total_portfolio_var
    return risk_contrib

def target_risk_contributions(target_risk, cov):
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def msd_risk(weights, target_risk, cov):
        w_contribs = risk_contribution(weights, cov)
        return ((w_contribs-target_risk)**2).sum()
    
    weights = minimize(msd_risk, init_guess,
                       args=(target_risk, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x

def equal_risk_contributions(cov):
    n = cov.shape[0]
    return target_risk_contributions(target_risk=np.repeat(1/n,n), cov=cov)


def optimale_rp_vægte(afkast):
    n_aktiver = afkast.shape[1]
    optimale_rul_rp_vægte = pd.DataFrame().reindex_like(afkast)
    optimale_rul_rp_vægte.columns = ['Vægt obl.', 'Vægt akt.']
    datoer = afkast.index
    n_steps = len(datoer)
    rp_vægte = 0
    
    for step in range(n_steps):
        if step >= 60:
            cov_rul = afkast.iloc[step-60:step].cov()
            rp_vægte = target_risk_contributions(1/n_aktiver, cov_rul)
        
        optimale_rul_rp_vægte.iloc[step] = rp_vægte      
        
    return optimale_rul_rp_vægte.iloc[60:n_steps]




#------------------------------- Risk Parity Levered -------------------------------------------
def target_risk_lev(target_vol, cov):
    
    n = cov.shape[0] # n is the number of assets
    init_guess = np.repeat(1/n, n) # Initial guess is 1/n per asset
    bounds = ((-5.0, 5.0),) * n # an N-tuple of 2-tuples!
    
    #Constraint
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights)-1
    }    
    
    # Objective function
    def objective(weights, target_vol, cov):
        portfolio_vol = (weights.T @ cov @ weights)**0.5
        return ((portfolio_vol - target_vol)**2)
    
    weights = minimize(objective, init_guess, 
                       args=(target_vol, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,), 
                       bounds=bounds)
            
    return weights.x


def target_risk(samlet_afkast, target_volatilitet):

    vægte = pd.DataFrame(index=samlet_afkast.index, columns=np.arange(2))
    vægte.columns = ['Vægt RP ugearet.', 'Vægt Lånerente']
    datoer = samlet_afkast.index
    n_steps = len(datoer)
    rp_g_vægte = 0
    
    for step in range(n_steps):
        if step >= 60:
            cov_rul = samlet_afkast.iloc[step-60:step].cov()*12
            rp_g_vægte = target_risk_lev(target_volatilitet, cov_rul)
        
        vægte.iloc[step] = rp_g_vægte      
        
    return vægte.iloc[60:n_steps]



#------------------------------- Efficient rand med 2 aktiver-------------------------

def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns



def portfolio_vol(weights, covmat):
    """
    Weights -> Vol
    """
    return (weights.T @ covmat @ weights)**0.5



def plot_ef2(n_points, er, cov, style=".-"):
    """
    Plots the 2 asset efficient frontier
    """
    if er.shape[0] != 2 or er.shape[0] !=2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    return ef.plot.line(x="Volatility", y="Returns", style=style)

#------------------------------------------------------------------------------------