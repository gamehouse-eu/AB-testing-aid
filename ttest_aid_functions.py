
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, ttest_ind
from statsmodels.stats.power import tt_ind_solve_power
from sklearn.model_selection import train_test_split
from scipy.stats import kstest,uniform
from statistics import stdev
from scipy import stats




def minimumDetectableEffect(sample):
    
    """
    This function takes a sample and returns the minimum detectable effect.
    The minimum detectable effect is the minimum difference in the mean of the
    sample that can be detected with 80% power.
    
    Parameters
    ----------
    sample : array_like
        The sample for which the minimum detectable effect is to be calculated.
        
    Returns
    -------
    mde : float
        The minimum detectable effect.
        
    Examples
    --------
    >>> minimumDetectableEffect(np.array([1,2,3,4,5,6,7,8,9,10]))
    0.3133
    """
    
    es = tt_ind_solve_power(effect_size=None, nobs1=sample.shape[0]//2,
                                     alpha=0.05, power=0.80, ratio=1.0, alternative='two-sided')
    m = sample.mean()
    s = stdev(sample)
    x = m - es*s
    mde = round(abs((x-m)/m),4)
    print('mean',m)
    print('minimum detectable effect', mde)
    print('+/-', mde*m)
    return abs((x-m)/m)



def AAtests(sample, no_tests = 1000):
    
    """
    This function performs a series of tests on the p-values of the t-test
    between two random samples of the same size from the same distribution.
    The tests are:
    1. The percentage of times the p-value is <=5%
    2. A chi-squared test on the 0.5 tail
    3. A Kolmogorov-Smirnov test on the uniformity assumption
    
    It also plots the distribution of the p-values and the distribution of the means of the two samples.
    
    Parameters
    ----------
    sample : array_like
        The sample from which the two random samples are drawn.
    no_tests : int
        The number of tests to perform.
        
    Returns
    -------
    dfPv : DataFrame
        A DataFrame containing the p-values of the t-test.
    dfM : DataFrame
        A DataFrame containing the mean values of the two random samples.
    """

    dfPv = []
    dfM = []
    for i in range(0,no_tests):
        train,test = train_test_split(sample,test_size = 0.5)
        dfPv.append(ttest_ind(train,test).pvalue)
        dfM.append(train.mean())
        dfM.append(test.mean())
    dfPv = pd.DataFrame(dfPv)
    dfPv.columns = ['p-values']
    print('Percentage of times the p-values was <=5%:',(dfPv['p-values']<=0.05).mean())
    
    f_obs = (dfPv<=0.05).value_counts()
    chiRes = chisquare(f_obs, f_exp=[0.95*no_tests,0.05*no_tests], ddof=0, axis=0)
    print('Chi-squared test on the 0.5 tail (should be high enough):',chiRes.pvalue)
    
    dfPv.hist()
    print('P-value of uniformity assumption (should be  high):',kstest(dfPv['p-values'], uniform.cdf).pvalue)
    
    dfM = pd.DataFrame(dfM)
    dfM.columns = ['mean values distribution']
    
    ((dfM-dfM.mean())/dfM.std()).hist(density=1, bins= 20 if no_tests <=1000 else 60)
    x = np.linspace(0 - 4,  4, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), color='red')
    plt.show()

    return dfPv, dfM


