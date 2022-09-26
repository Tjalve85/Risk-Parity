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
    # convert the annual riskfree rate to per period
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def sharpe_ratio_rf(r, riskfree_rate, periods_per_year):
    excess_ret = r.subtract(riskfree_rate, axis=0)
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol


def drawdown(return_series: pd.Series):
    wealth_index = 100*(1+return_series).cumprod()
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

#-------- Validering af risikobidrag -----------------------------------
def risk_contribution(weights,cov):
    total_portfolio_vol =  (weights.T @ cov @ weights)**0.5
    
    # Marginal contribution of each asset
    marginal_contrib = cov@weights
    
    # Risk contribution for each asset
    risk_contrib = np.multiply(marginal_contrib,weights.T)/total_portfolio_vol
    risk_contrib_pct = risk_contrib/total_portfolio_vol
    return risk_contrib_pct

def r_bidrag(w, cov):
    datoer = w.index
    n_steps = len(datoer)
    risiko_b = pd.DataFrame().reindex_like(w)
    risiko_b.columns = ['RB akt.', 'RB obl.']
    rb = 0
    for step in range(n_steps):
        # For cov ganges step med antal aktiver, for at få alle rækker i kovariansmatricen med
        rb = risk_contribution(w.iloc[step].ravel(), cov.iloc[step*2:step*2+2]) 
        risiko_b.iloc[step] = rb.ravel()      
    return risiko_b

def r_bidrag_aoc(w, cov):
    datoer = w.index
    n_steps = len(datoer)
    risiko_b = pd.DataFrame().reindex_like(w)
    risiko_b.columns = ['RB akt.', 'RB obl.', 'RB com.']
    rb = 0
    for step in range(n_steps):
        # For cov ganges step med antal aktiver, for at få alle rækker i kovariansmatricen med
        rb = risk_contribution(w.iloc[step].ravel(), cov.iloc[step*3:step*3+3]) 
        risiko_b.iloc[step] = rb.ravel()      
    return risiko_b

def r_bidrag_aocs(w, cov):
    datoer = w.index
    n_steps = len(datoer)
    risiko_b = pd.DataFrame().reindex_like(w)
    risiko_b.columns = ['RB akt.', 'RB obl.', 'RB com.', 'RB style']
    rb = 0
    for step in range(n_steps):
        # For cov ganges step med antal aktiver, for at få alle rækker i kovariansmatricen med
        rb = risk_contribution(w.iloc[step].ravel(), cov.iloc[step*4:step*4+4]) 
        risiko_b.iloc[step] = rb.ravel()      
    return risiko_b


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
    n_steps = len(datoer)
    rp_vægte = 0
            
    for step in range(n_steps):
        if step >= 252 and rebalancerings_dage.iloc[step,0] == 'Rebalancering':
            cov_rul = afkast.iloc[step-252:step].cov()
            rp_vægte = target_risk_contributions(1/n_aktiver, cov_rul)
        else:
            rp_vægte = None
        
        optimale_rul_rp_vægte.iloc[step] = rp_vægte      
        
    return optimale_rul_rp_vægte.iloc[252:n_steps]


def rebalancering_rp_ug(afkast, vægte, startværdi=100):
    datoer = afkast.index
    n_steps = len(datoer)
    v_1 = vægte.iloc[0,0]*startværdi
    v_2 = vægte.iloc[0,1]*startværdi
    v_3 = vægte.iloc[0,2]*startværdi
    v_4 = vægte.iloc[0,3]*startværdi
    v_5 = vægte.iloc[0,4]*startværdi
    v_6 = vægte.iloc[0,5]*startværdi
    v_7 = vægte.iloc[0,6]*startværdi
    v_pf = v_1+v_2+v_3+v_4+v_5+v_6+v_7

    wealth_historie = pd.DataFrame(index=afkast.index, columns=np.arange(1))
    wealth_historie.columns = ['Wealth']
    for step in range(n_steps):   
        if pd.isnull(vægte.iloc[step,0]) == False:
            v_1 = vægte.iloc[step,0]*v_pf*(1+afkast.iloc[step,0])
            v_2 = vægte.iloc[step,1]*v_pf*(1+afkast.iloc[step,1])
            v_3 = vægte.iloc[step,2]*v_pf*(1+afkast.iloc[step,2])
            v_4 = vægte.iloc[step,3]*v_pf*(1+afkast.iloc[step,3])
            v_5 = vægte.iloc[step,4]*v_pf*(1+afkast.iloc[step,4])
            v_6 = vægte.iloc[step,5]*v_pf*(1+afkast.iloc[step,5])
            v_7 = vægte.iloc[step,6]*v_pf*(1+afkast.iloc[step,6])
            v_pf = v_1+v_2+v_3+v_4+v_5+v_6+v_7
        
        else:

            v_1 = v_1*(1+afkast.iloc[step,0])
            v_2 = v_2*(1+afkast.iloc[step,1])
            v_3 = v_3*(1+afkast.iloc[step,2])
            v_4 = v_4*(1+afkast.iloc[step,3])
            v_5 = v_5*(1+afkast.iloc[step,4])
            v_6 = v_6*(1+afkast.iloc[step,5])
            v_7 = v_7*(1+afkast.iloc[step,6])
            v_pf = v_1+v_2+v_3+v_4+v_5+v_6+v_7

        wealth_historie.iloc[step] = v_pf 
    return wealth_historie

#------------------------------- Risk Parity Levered -------------------------------------------
def target_risk_lev(target_vol, cov):
    
    n = cov.shape[0] # n is the number of assets
    init_guess = np.repeat(1/n, n) # Initial guess is 1/n per asset
    bounds = ((-10.0, 10.0),) * n # an N-tuple of 2-tuples!
    
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
    
    rebalancerings_dage = pd.DataFrame().reindex_like(samlet_afkast)
    labels_rp = list(rebalancerings_dage.columns)
    mnd_data =  rebalancerings_dage.resample('BMS').first()
    mnd_data[:]='Rebalancering'
    rebalancerings_dage[labels_rp] = mnd_data[labels_rp]
    rebalancerings_dage.iloc[252,0:]= 'Rebalancering'
     
    datoer = samlet_afkast.index
    n_steps = len(datoer)
    rp_g_vægte = 0
    
    for step in range(n_steps):
        if step >= 252 and rebalancerings_dage.iloc[step,0] == 'Rebalancering':
            cov_rul = samlet_afkast.iloc[step-252:step].cov()*252
            rp_g_vægte = target_risk_lev(target_volatilitet, cov_rul)
        
        vægte.iloc[step] = rp_g_vægte      
        
    return vægte.iloc[252:n_steps]



def gearing(rebalanceringsdage, gearings_faktor):
    gearing = pd.DataFrame().reindex_like(gearings_faktor)
    rebalancerings_dage = rebalanceringsdage
    rebalancerings_dage.iloc[0,0:]= 'Rebalancering'  
    datoer = gearings_faktor.index
    n_steps = len(datoer)
    for step in range(n_steps):
        if rebalancerings_dage.iloc[step][0] == 'Rebalancering':
            gearing_faktor = gearings_faktor.iloc[step]
        else:
            gearing_faktor = gearing.iloc[step-1]
        gearing.iloc[step] = gearing_faktor
    return gearing

#------------------------------- Efficient rand med 2 aktiver-------------------------

def portfolio_return(weights, returns):
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov, style=".-"):
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

#------------------------------------------- Stats til Delanalyse 2 -----------------------------------------

def stats_d2(r, riskfree_rate):
    ann_r = annualize_rets(r, periods_per_year=252)
    ann_vol = annualize_vol(r, periods_per_year=252)
    ann_sr = sharpe_ratio_rf(r, riskfree_rate, periods_per_year=252)
    dd = r.aggregate(lambda r: drawdown(r).Drawdown.min())
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_vol,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
    })

#------------------------------------------- CPPI -----------------------------------------

def cppi(risky_r, safe_r=None, m=3, start=100, floor=0.8):

    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    
    rebalancerings_dage = pd.DataFrame().reindex_like(risky_r)
    labels_mv = list(rebalancerings_dage.columns)
    mnd_data =  rebalancerings_dage .resample('BMS').first()
    mnd_data[:]='Rebalancering'
    rebalancerings_dage[labels_mv] = mnd_data[labels_mv]
    rebalancerings_dage.iloc[0,0:]= 'Rebalancering'
        
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floor_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        peak = np.maximum(peak, account_value)
                
        if rebalancerings_dage.iloc[step,0] == 'Rebalancering':      
            floor_value = peak*floor
            cushion = (account_value - floor_value)/account_value 
            risky_w = m*cushion
            risky_w = np.minimum(risky_w, 1)
            risky_w = np.maximum(risky_w, 0)
            safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value  
        floor_history.iloc[step] = floor_value
            
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor_history": floor_history,
        "risky_r":risky_r,
        "safe_r": safe_r
    }
    return backtest_result

#------------------------------------ Minimum Varians portefølje og Tangent portefølje------------------------------------------------
from scipy.optimize import minimize

def msr(riskfree_rate, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe(weights, riskfree_rate, er, cov):
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    weights = minimize(neg_sharpe, init_guess,
                       args=(riskfree_rate, er, cov), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,),
                       bounds=bounds)
    return weights.x


def gmv(cov):
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def optimale_mv_vægte(rebalanceringsdage,afkast):   
    n_aktiver = afkast.shape[1]
    optimale_rul_mv_vægte = pd.DataFrame().reindex_like(afkast)
    rebalancerings_dage = rebalanceringsdage
    rebalancerings_dage.iloc[252,0:]= 'Rebalancering'   
    datoer = afkast.index
    n_steps = len(datoer)
    mv_vægte = 0           
    for step in range(n_steps):
        if step >= 252 and rebalancerings_dage.iloc[step,0] == 'Rebalancering':
            cov_rul = afkast.iloc[step-252:step].cov()
            mv_vægte = gmv(cov_rul)
        else:
            mv_vægte = None        
        optimale_rul_mv_vægte.iloc[step] = mv_vægte              
    return optimale_rul_mv_vægte.iloc[252:n_steps]


def optimale_tangent_vægte(rebalanceringsdage, afkast, risikofri_rente):      
    n_aktiver = afkast.shape[1]
    optimale_rul_tangent_vægte = pd.DataFrame().reindex_like(afkast)
    rebalancerings_dage = rebalanceringsdage
    rebalancerings_dage.iloc[252,0:]= 'Rebalancering'  
    datoer = afkast.index
    n_steps = len(datoer)
    tangent_vægte = 0            
    for step in range(n_steps):
        if step >= 252 and rebalancerings_dage.iloc[step,0] == 'Rebalancering':
            cov_rul = afkast.iloc[step-252:step].cov()
            forventet_afkast = afkast.iloc[step-252:step].mean()
            tangent_vægte = msr(risikofri_rente.iloc[step,:], forventet_afkast, cov_rul)
        else:
            tangent_vægte = None        
        optimale_rul_tangent_vægte.iloc[step] = tangent_vægte              
    return optimale_rul_tangent_vægte.iloc[252:n_steps]

#------------------------------------ Efficient rand med n aktiver ------------------------------------------------
def minimize_vol(target_return, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x


def optimal_weights(n_points, er, cov):
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights




def plot_ef(n_points, er, cov, style='.-', legend=False, show_cml=False, riskfree_rate=0, show_ew=False, show_gmv=False, figsize=(12,6)):
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets, 
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend, figsize=(12,6))
    if show_cml:
        ax.set_xlim(left = 0)
        # get MSR
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # add CML
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # add EW
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o', markersize=10)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # add EW
        ax.plot([vol_gmv], [r_gmv], color='midnightblue', marker='o', markersize=10)
        
        return ax

    
#------------------------------------ Rebalancering ---------------------------------------------

# 60/40 portefølje - denne skal slettes eller opdateres

def rebalancering_ao(afkast, vægte, startværdi=100):
    datoer = afkast.index
    n_steps = len(datoer)
    v_1 = vægte.iloc[0,0]*startværdi
    v_2 = vægte.iloc[0,1]*startværdi
    v_pf = v_1+v_2
    
    wealth_historie = pd.DataFrame(index=afkast.index, columns=np.arange(1))
    wealth_historie.columns = ['Wealth']
    for step in range(n_steps):   
        if pd.isnull(vægte.iloc[step,0]) == False:
            v_1 = vægte.iloc[step,0]*v_pf*(1+afkast.iloc[step,0])
            v_2 = vægte.iloc[step,1]*v_pf*(1+afkast.iloc[step,1])
            v_pf = v_1+v_2
        else:
            v_1 = v_1*(1+afkast.iloc[step,0])
            v_2 = v_2*(1+afkast.iloc[step,1])            
            v_pf = v_1+v_2

        wealth_historie.iloc[step] = v_pf 
    return wealth_historie
# -----------------Ligevægtet, MV og tangent ----------------------
def rebalancering(afkast, vægte, startværdi=100):
    
    datoer = afkast.index
    n_steps = len(datoer)
    v_1 = vægte.iloc[0,0]*startværdi
    v_2 = vægte.iloc[0,1]*startværdi
    v_3 = vægte.iloc[0,2]*startværdi
    v_4 = vægte.iloc[0,3]*startværdi
    v_5 = vægte.iloc[0,4]*startværdi
    v_6 = vægte.iloc[0,5]*startværdi
    v_7 = vægte.iloc[0,6]*startværdi
    v_8 = vægte.iloc[0,7]*startværdi
    v_pf = v_1+v_2+v_3+v_4+v_5+v_6+v_7+v_8

    wealth_historie = pd.DataFrame(index=afkast.index, columns=np.arange(1))
    wealth_historie.columns = ['Wealth']
    for step in range(n_steps):   
        if pd.isnull(vægte.iloc[step,0]) == False:
            v_1 = vægte.iloc[step,0]*v_pf*(1+afkast.iloc[step,0])
            v_2 = vægte.iloc[step,1]*v_pf*(1+afkast.iloc[step,1])
            v_3 = vægte.iloc[step,2]*v_pf*(1+afkast.iloc[step,2])
            v_4 = vægte.iloc[step,3]*v_pf*(1+afkast.iloc[step,3])
            v_5 = vægte.iloc[step,4]*v_pf*(1+afkast.iloc[step,4])
            v_6 = vægte.iloc[step,5]*v_pf*(1+afkast.iloc[step,5])
            v_7 = vægte.iloc[step,6]*v_pf*(1+afkast.iloc[step,6])
            v_8 = vægte.iloc[step,7]*v_pf*(1+afkast.iloc[step,7])
            v_pf = v_1+v_2+v_3+v_4+v_5+v_6+v_7+v_8
        
        else:
            v_1 = v_1*(1+afkast.iloc[step,0])
            v_2 = v_2*(1+afkast.iloc[step,1])
            v_3 = v_3*(1+afkast.iloc[step,2])
            v_4 = v_4*(1+afkast.iloc[step,3])
            v_5 = v_5*(1+afkast.iloc[step,4])
            v_6 = v_6*(1+afkast.iloc[step,5])
            v_7 = v_7*(1+afkast.iloc[step,6])
            v_8 = v_8*(1+afkast.iloc[step,7])           
            v_pf = v_1+v_2+v_3+v_4+v_5+v_6+v_7+v_8
        v1_historie.iloc[step] = v_1
        v2_historie.iloc[step] = v_2
        v3_historie.iloc[step] = v_3
        v4_historie.iloc[step] = v_4
        v5_historie.iloc[step] = v_5
        v6_historie.iloc[step] = v_6
        v7_historie.iloc[step] = v_7
        v8_historie.iloc[step] = v_8
        wealth_historie.iloc[step] = v_pf 
    return wealth_historie



# ------------------------------- Formlertil gearing og rebalancering ----------


def opt_vægte_til_rebalancering(rebalanceringsdage, opt_vægte_daglig):
    rebalancerings_dage = rebalanceringsdage
    opt_vægte = pd.DataFrame().reindex_like(opt_vægte_daglig)
    rebalancerings_dage.iloc[0,0:]= 'Rebalancering'  
    datoer = opt_vægte_daglig.index
    for dato in datoer:
        if rebalancerings_dage.loc[dato][0] == 'Rebalancering':
            vægte = opt_vægte_daglig.loc[dato]
        else:
            vægte = None
        opt_vægte.loc[dato] = vægte
    return opt_vægte


def rebalancering_aoc_pf(afkast, vægte, startværdi=100):
    datoer = afkast.index
    n_steps = len(datoer)
    v_1 = vægte.iloc[0,0]*startværdi
    v_2 = vægte.iloc[0,1]*startværdi
    v_3 = vægte.iloc[0,2]*startværdi
    v_pf = v_1+v_2+v_3
    
    wealth_historie = pd.DataFrame(index=afkast.index, columns=np.arange(1))
    wealth_historie.columns = ['Wealth']
    
    v1_historie = pd.DataFrame(index=afkast.index, columns=np.arange(1))
    v1_historie.columns = ['Wealth']   
    
    v2_historie = pd.DataFrame(index=afkast.index, columns=np.arange(1))
    v2_historie.columns = ['Wealth'] 
    
    v3_historie = pd.DataFrame(index=afkast.index, columns=np.arange(1))
    v3_historie.columns = ['Wealth']
    
    for step in range(n_steps):   
        if pd.isnull(vægte.iloc[step,0]) == False:
            v_1 = vægte.iloc[step,0]*v_pf*(1+afkast.iloc[step,0])
            v_2 = vægte.iloc[step,1]*v_pf*(1+afkast.iloc[step,1])
            v_3 = vægte.iloc[step,2]*v_pf*(1+afkast.iloc[step,2])
            v_pf = v_1+v_2+v_3
        else:
            v_1 = v1_historie.iloc[step-1]*(1+afkast.iloc[step,0])
            v_2 = v2_historie.iloc[step-1]*(1+afkast.iloc[step,1])
            v_3 = v3_historie.iloc[step-1]*(1+afkast.iloc[step,2]) 
            v_pf = v_1+v_2+v_3
        v1_historie.iloc[step] = v_1
        v2_historie.iloc[step] = v_2
        v3_historie.iloc[step] = v_3

        wealth_historie.iloc[step] = v_pf 
    return wealth_historie

def rebalancering_aocs_pf(afkast, vægte, startværdi=100):
    datoer = afkast.index
    n_steps = len(datoer)
    v_1 = vægte.iloc[0,0]*startværdi
    v_2 = vægte.iloc[0,1]*startværdi
    v_3 = vægte.iloc[0,2]*startværdi
    v_4 = vægte.iloc[0,3]*startværdi
    v_pf = v_1+v_2+v_3+v_4    
    wealth_historie = pd.DataFrame(index=afkast.index, columns=np.arange(1))
    wealth_historie.columns = ['Wealth']    
    v1_historie = pd.DataFrame(index=afkast.index, columns=np.arange(1))
    v1_historie.columns = ['Wealth']       
    v2_historie = pd.DataFrame(index=afkast.index, columns=np.arange(1))
    v2_historie.columns = ['Wealth']     
    v3_historie = pd.DataFrame(index=afkast.index, columns=np.arange(1))
    v3_historie.columns = ['Wealth']        
    v4_historie = pd.DataFrame(index=afkast.index, columns=np.arange(1))
    v4_historie.columns = ['Wealth']     
    for step in range(n_steps):   
        if pd.isnull(vægte.iloc[step,0]) == False:
            v_1 = vægte.iloc[step,0]*v_pf*(1+afkast.iloc[step,0])
            v_2 = vægte.iloc[step,1]*v_pf*(1+afkast.iloc[step,1])
            v_3 = vægte.iloc[step,2]*v_pf*(1+afkast.iloc[step,2])
            v_4 = vægte.iloc[step,3]*v_pf*(1+afkast.iloc[step,3])
            v_pf = v_1+v_2+v_3+v_4
        else:
            v_1 = v1_historie.iloc[step-1]*(1+afkast.iloc[step,0])
            v_2 = v2_historie.iloc[step-1]*(1+afkast.iloc[step,1])
            v_3 = v3_historie.iloc[step-1]*(1+afkast.iloc[step,2]) 
            v_4 = v4_historie.iloc[step-1]*(1+afkast.iloc[step,3]) 
            v_pf = v_1+v_2+v_3+v_4
        v1_historie.iloc[step] = v_1
        v2_historie.iloc[step] = v_2
        v3_historie.iloc[step] = v_3
        v4_historie.iloc[step] = v_4
        wealth_historie.iloc[step] = v_pf 
    return wealth_historie


def cppi(rebalanceringsdage, risky_r, safe_r=None, m=3, start=100, floor=0.8):
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = account_value
    rebalancerings_dage = rebalanceringsdage
    rebalancerings_dage.iloc[0,0:]= 'Rebalancering'
    account_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    floor_history = pd.DataFrame().reindex_like(risky_r)
    
    for step in range(n_steps):
        peak = np.maximum(peak, account_value)
                
        if rebalancerings_dage.iloc[step,0] == 'Rebalancering':      
            floor_value = peak*floor
            cushion = (account_value - floor_value)/account_value 
            risky_w = m*cushion
            risky_w = np.minimum(risky_w, 1)
            risky_w = np.maximum(risky_w, 0)
            safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value  
        floor_history.iloc[step] = floor_value
            
    risky_wealth = start*(1+risky_r).cumprod()
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor_history": floor_history,
        "risky_r":risky_r,
        "safe_r": safe_r
    }
    return backtest_result


# -----AOC ERC RP ---------------- Opdateret ud fra Bai et al, formel (13) - med kovarianser som input
from scipy.optimize import minimize

# Funktion der finder de optimale Risk Parity vægte
def optimale_rp_vægte_aoc(cov):
    n_aktiver = cov.shape[1]
    date_index = cov.index.get_level_values(0)
    datoindex = []
    for date, cov_date in cov.groupby(date_index):
        datoindex.append(date)    
    labels_rp = ['Aktie portefølje', 'Obligations portefølje', 'Commodities']    
    optimale_rul_rp_vægte = pd.DataFrame(columns = ['Aktie portefølje', 'Obligations portefølje', 'Commodities'], index=datoindex)    
    rebalancerings_dage = pd.DataFrame(columns = ['Aktie portefølje', 'Obligations portefølje', 'Commodities'], index=datoindex)    
    
    # Her laves en DataFrame med værdier svarende til index nummer, så vi kan lokalisere
    # værdien der skal bruges med .iloc. Dette bruges i for-loopet til at finde værdien dagen før,
    # der skal indgå som gæt
    step = pd.DataFrame(columns = ['step'], index=datoindex)  
    step['step'] = np.arange(len(datoindex))    
   
    for date, cov_date in cov.groupby(date_index):
        cov_mat = cov_date.reset_index('Date', drop=True).to_numpy()        
        if step['step'].loc[date] == 0:
            gæt = 1/ n_aktiver * np.ones(n_aktiver)           
        else:
            gæt = optimale_rul_rp_vægte.iloc[step.loc[date]-1]       
        rp_vægte = target_risk_contributions(cov_mat, gæt)               
        optimale_rul_rp_vægte.loc[date] = rp_vægte       
    return optimale_rul_rp_vægte

# The object function (13) in (Xi Bai et al.)
def orthants_object_function(x, beta, risk_budget, cov_matrix):
    res = 0.5 * np.dot(np.dot(x, cov_matrix), x) - sum(risk_budget * np.log(beta * x))    
    return res

def target_risk_contributions(cov_matrix, initial_guess):
    n = cov_matrix.shape[1] # antal aktiver   
    beta = np.ones(n)
    bounds = tuple((0, None) if b > 0 else (None, 0) for b in beta)
    risk_budget = np.ones(len(beta))/n
    fconst = lambda w: np.sum(w) - 1
    cons = ({'type': 'eq', 'fun': fconst})
    res = minimize(lambda x: orthants_object_function(x, beta, risk_budget, cov_matrix),
                   initial_guess,
                   tol=10 ** -16,
                   bounds=bounds,
                   #constraints=cons,
                   options={'maxiter': 10000})  
    return res.x/np.sum(res.x)

# ------AOC 40/40/20 RP --------------- Opdateret ud fra Bai et al, formel (13) - med kovarianser som input
from scipy.optimize import minimize

# Funktion der finder de optimale Risk Parity vægte
def optimale_rp_vægte_aoc_404020(cov):
    n_aktiver = cov.shape[1]
    date_index = cov.index.get_level_values(0)
    datoindex = []
    for date, cov_date in cov.groupby(date_index):
        datoindex.append(date)    
    labels_rp = ['Aktie portefølje', 'Obligations portefølje', 'Commodities']    
    optimale_rul_rp_vægte = pd.DataFrame(columns = ['Aktie portefølje', 'Obligations portefølje', 'Commodities'], index=datoindex)    
    rebalancerings_dage = pd.DataFrame(columns = ['Aktie portefølje', 'Obligations portefølje', 'Commodities'], index=datoindex)        
    # Her laves en DataFrame med værdier svarende til index nummer, så vi kan lokalisere
    # værdien der skal bruges med .iloc. Dette bruges i for-loopet til at finde værdien dagen før,
    # der skal indgå som gæt
    step = pd.DataFrame(columns = ['step'], index=datoindex)  
    step['step'] = np.arange(len(datoindex))   
    for date, cov_date in cov.groupby(date_index):
        cov_mat = cov_date.reset_index('Date', drop=True).to_numpy()        
        if step['step'].loc[date] == 0:
            gæt = 1/ n_aktiver * np.ones(n_aktiver)           
        else:
            gæt = optimale_rul_rp_vægte.iloc[step.loc[date]-1]       
        rp_vægte = target_risk_contributions_404020(cov_mat, gæt)               
        optimale_rul_rp_vægte.loc[date] = rp_vægte       
    return optimale_rul_rp_vægte

# The object function (13) in (Xi Bai et al.)
def orthants_object_function(x, beta, risk_budget, cov_matrix):
    res = 0.5 * np.dot(np.dot(x, cov_matrix), x) - sum(risk_budget * np.log(beta * x))    
    return res

def target_risk_contributions_404020(cov_matrix, initial_guess):
    n = cov_matrix.shape[1] # antal aktiver
   
    beta = np.ones(n)
    bounds = tuple((0, None) if b > 0 else (None, 0) for b in beta)
    risk_budget = np.array([0.4, 0.4, 0.2])
    fconst = lambda w: np.sum(w) - 1
    cons = ({'type': 'eq', 'fun': fconst})

    res = minimize(lambda x: orthants_object_function(x, beta, risk_budget, cov_matrix),
                   initial_guess,
                   tol=10 ** -16,
                   bounds=bounds,
                   #constraints=cons,
                   options={'maxiter': 10000})   
    return res.x/np.sum(res.x)

# ----AOCS RP 35/35/15/15---------- Opdateret ud fra Bai et al, formel (13) - med kovarianser som input
from scipy.optimize import minimize

# Funktion der finder de optimale Risk Parity vægte
def optimale_rp_vægte_aocs_35351515(cov):
    n_aktiver = cov.shape[1]
    date_index = cov.index.get_level_values(0)
    datoindex = []
    for date, cov_date in cov.groupby(date_index):
        datoindex.append(date)
        
    labels_rp = ['Aktie portefølje', 'Obligations portefølje', 'Commodities', 'Style']    
    optimale_rul_rp_vægte = pd.DataFrame(columns = ['Aktie portefølje', 'Obligations portefølje', 'Commodities', 'Style'], index=datoindex)    
    rebalancerings_dage = pd.DataFrame(columns = ['Aktie portefølje', 'Obligations portefølje', 'Commodities', 'Style'], index=datoindex)    
    
    step = pd.DataFrame(columns = ['step'], index=datoindex)  
    step['step'] = np.arange(len(datoindex))   
   
    for date, cov_date in cov.groupby(date_index):
        cov_mat = cov_date.reset_index('Date', drop=True).to_numpy()        
        if step['step'].loc[date] == 0:
            gæt = 1/ n_aktiver * np.ones(n_aktiver)           
        else:
            gæt = optimale_rul_rp_vægte.iloc[step.loc[date]-1]       
        rp_vægte = target_risk_contributions_35351515(cov_mat, gæt)              
        optimale_rul_rp_vægte.loc[date] = rp_vægte
       
    return optimale_rul_rp_vægte

# The object function (13) in (Xi Bai et al.)
def orthants_object_function(x, beta, risk_budget, cov_matrix):
    res = 0.5 * np.dot(np.dot(x, cov_matrix), x) - sum(risk_budget * np.log(beta * x))    
    return res

def target_risk_contributions_35351515(cov_matrix, initial_guess):
    n = cov_matrix.shape[1] # antal aktiver
   
    beta = np.ones(n)
    bounds = tuple((0, None) if b > 0 else (None, 0) for b in beta)
    risk_budget = np.array([0.35, 0.35, 0.15, 0.15])
    fconst = lambda w: np.sum(w) - 1
    cons = ({'type': 'eq', 'fun': fconst})

    res = minimize(lambda x: orthants_object_function(x, beta, risk_budget, cov_matrix),
                   initial_guess,
                   tol=10 ** -16,
                   bounds=bounds,
                   #constraints=cons,
                   options={'maxiter': 10000})  
 
    return res.x/np.sum(res.x)
#----------------------