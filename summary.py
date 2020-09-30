import os
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import scipy.stats as st
import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import rank_filter
import statsmodels.api as sm

# TO find the best fit distribution of histogram
def best_fit_distribution(data, bins=200, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    DISTRIBUTIONS = [
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
        st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return (best_distribution.name, best_params)


# To explain trends if there is any time line in graph
def predictTrend(data,ylabels,xlabels,bar_type,x_title,y_title):
    Summ=''
    for i in range(len(ylabels)):
        hgt = data[:,i]
        local_max = argrelextrema(hgt, np.greater)
        local_min = argrelextrema(hgt, np.less)
        order = np.array([0] * len(xlabels))
        order[local_min] = -1
        order[local_max] = 1

        if ylabels[0]=='Y' and len(ylabels)==1:
            trend_str=". The Y axis value"
            if y_title != '_':
                trend_str=". The "+y_title
        else:
            trend_str=". The "+ylabels[i]
            if y_title != '_':
                trend_str=". The "+y_title+" of "+ylabels[i]



        if list(order)==[0]*len(xlabels):
            if(int(hgt[0])<int(hgt[1])):
                trend_str += " has an overall increasing trend"
                if( int(hgt[len(hgt)-2])>int(hgt[len(hgt)-1]) ):
                    trend_str += " till "+str(xlabels[len(xlabels)-2])+" and ends with a drop in "+str(xlabels[len(xlabels)-1])
                else:
                    trend_str += " from "+str(xlabels[0])+" to "+str(xlabels[len(xlabels)-1])
            elif(int(hgt[0])>int(hgt[1])):
                trend_str += " has an overall decreasing trend"
                if( int(hgt[len(hgt)-2])<int(hgt[len(hgt)-1]) ):
                    trend_str += " till "+str(xlabels[len(xlabels)-2])+" and ends with a peak in "+str(xlabels[len(xlabels)-1])
                else:
                    trend_str += " from "+str(xlabels[0])+" to "+str(xlabels[len(xlabels)-1])
            else:
                if(int(hgt[0])!=int(hgt[len(hgt)-1])):
                    trend_str += " is uniform with "+str(int(hgt[0]))+" till "+str(xlabels[len(xlabels)-2])+" and finally ends with "+str(int(hgt[len(hgt)-1]))+" in "+str(xlabels[len(xlabels)-1])
                else:
                    trend_str += " is uniform with "+str(int(hgt[0]))+" throughout the entire period"
        else:
            trend_str += " starts with "+str(int(hgt[0]))+" in "+str(xlabels[0])+" then "
            j=1
            while j<len(order):
                if order[j]==-1:
                    if list(order[:j])==[0]*j:
                        trend_str += "declines till "+str(xlabels[j])+", followed by "
                    else:
                        trend_str += "a decreasing trend till "+str(xlabels[j])+", "
                elif order[j]==1:
                    if list(order[:j])==[0]*j:
                        trend_str += "increases till "+str(xlabels[j])+", followed by "
                    else :
                        trend_str += "an increasing trend till "+str(xlabels[j])+", "
                j+=1
            if(order[j-2]!=0):
                trend_str += "and finally ends with "+str(int(hgt[j-1]))+" in "+str(xlabels[j-1])
            else :
                if(hgt[j-1]<hgt[j-2]):
                    trend_str += "a decreasing trend till "+str(xlabels[j-1])+" the end"
                else:
                    trend_str += "an increasing trend till "+str(xlabels[j-1])+" the end"


        # speaking about trend of each group
        Summ += trend_str
    return Summ


def summaryGen(filename):
    image_name = os.path.basename(filename).split(".png")[0]
    path = os.path.dirname(filename)+'/'
    df = pd.read_csv(path+"data_"+image_name+".csv", sep=",", index_col=False)
    xlabel = (df.loc[ : , list(df)[0]]).values
    xlabels = []
    for i in xlabel:
        if isinstance(i, np.float64):
            xlabels += [int(i)]
        else :
            xlabels += [i]

    ylabels = list(df)[1:len(list(df))-4]

    data = (df.loc[ : , ylabels]).values
    x_title = df['x-title'][0]
    y_title = df['y-title'][0]
    title = df['title'][0]
    bar_type = df['bar_type'][0]


    if bar_type == 'Histogram':
        min_w = round(min(list(df['bin_width'])),2)
        freq = []
        for i in np.array(df.iloc[:,0:3]):
            freq+=[int(i[0])]*int(i[1])
        data = pd.Series(freq)
        mode_id = list(df['freq']).index(max(list(df['freq'])))
        if(title == '_'):
            Summ = 'The plot depicts a '+bar_type+' with the bins ranging from '
        else:
            Summ = 'The plot depicts a '+bar_type+' of '+title+'. The bins range from '
        best_fit_name, best_fit_params = best_fit_distribution(data, len(df), None)
        best_dist = getattr(st, best_fit_name)
        param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
        param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])

        Summ += str(round(df['bin_center'][0]))+' to '+str(round(df['bin_center'][len(df)-1]))+' with '+str(min_w)+' bin width. The mode of a histogram is '+str(round(df['bin_center'][mode_id]))+' with a frequency of '+str(int(round(df['freq'][mode_id])))+'. The frequency distribution of histogram is the '+best_fit_name+' with following parameters '+param_str+'.'

    else:
        Summ = 'The plot depicts a '+bar_type+' Graph'

        if title !='_':
            Summ += ' illustrating '+title

        if x_title != '_' and y_title != '_':
            Summ +='. The plot is between '+y_title+' on y-axis over '+x_title+' on the x-axis'
        elif y_title != '_':
            Summ +='. The plot is having '+y_title+' on y-axis'
        elif x_title != '_':
            Summ +='. The plot is having '+x_title+' on x-axis'

        # speaking about legend
        if bar_type != 'Vertical Simple Bar' and bar_type != 'Horizontal Simple Bar':
            Summ +=' for '
            for i in range(len(ylabels)-1):
                Summ += str(ylabels[i])+", "
            Summ += "and "+str(ylabels[i+1])

        if bar_type == 'Horizontal Simple Bar' or bar_type == 'Horizontal Grouped Bar' or bar_type == 'Horizontal Stacked Bar':
            temp = y_title
            y_title = x_title
            x_title = temp

        # To speak about trend in graph if it has ordered attributes, or It is already in sorted order
        trend_found = False
        # the attributes whose order of values is important
        order_attr = ['date','year','month','day','age']
        # check if any title has this attribute
        for i in order_attr :
            if (i in y_title.lower()) or (i in x_title.lower()):
                trend_found=True
                Summ += predictTrend(data,ylabels,xlabels,bar_type,x_title,y_title)
                break

        if not trend_found:
            if bar_type == 'Horizontal Simple Bar' or bar_type == 'Vertical Simple Bar':
                # Graph is already in sorted order
                if ((data[:,0]==sorted(data[:,0])).all() or (data[:,0]==sorted(data[:,0],reverse=True)).all()):
                    trend_found=True
                    Summ += predictTrend(data,ylabels,xlabels,bar_type,x_title,y_title)
                else:
                    dat, xlabels= zip(*sorted(zip(np.round(data[:,0].tolist(), decimals=2), xlabels), reverse=True))
                    if x_title != '_':
                        if y_title == '_':
                            y_title = 'value'
                        Summ += '. The '+x_title+' with the highest '+y_title+' '+str(dat[0])+' is \''+str(xlabels[0])+'\'. The '+x_title+' with the lowest '+y_title+' '+str(dat[len(dat)-1])+' is \''+str(xlabels[len(dat)-1])+'\'. The mean '+y_title+' of '+x_title+' is '+str(round(sum(dat)/len(dat),2))
                    else:
                        if y_title == '_':
                            y_title = 'value'
                        Summ += '. The highest '+y_title+' '+str(dat[0])+' is \''+str(xlabels[0])+'\'. The lowest '+y_title+' '+str(dat[len(dat)-1])+' is \''+str(xlabels[len(dat)-1])+'\'. The mean '+y_title+' is '+str(round(sum(dat)/len(dat),2))

            # For Catogeorical Graphs
            else:
                # To speak about x axis labels
                if x_title == '_':
                    if 'Vertical' in bar_type:
                        x_title = 'X-axis'
                    else :
                        x_title = 'Y-axis'
                if(isinstance(xlabels[0], str)):
                    Summ +='. The list of \''+x_title+'\' values is '
                    for i in range(len(xlabels)-1):
                        Summ += xlabels[i]+', '
                    Summ += 'and '+xlabels[i]
                else:
                    Summ += '. The range of \''+x_title+'\' values are '+str(xlabels[0])+' to '+str(xlabels[len(xlabels)-1])

                # To represent ranges of all groups
                for i in range(len(ylabels)):
                    Summ += '. The \''+str(ylabels[i])+'\' range from '+str(round(min(data[:,i]),2))+' to '+str(round(max(data[:,i]),2))+', with a standard deviation of '+str(round(np.std(data[:,i]),2))

                # Check for Correlation
                corr_mat = np.triu(df.iloc[:,1:len(list(df))-4].corr(method='spearman'), k=1)
                x,y=np.nonzero(abs(corr_mat)>0.5)
                for j in range(len(x)):
                    if corr_mat[x[j],y[j]]>0:
                        Summ += '. The categories \''+str(ylabels[x[j]])+"\' and \'"+str(ylabels[y[j]])+'\' are positively correlated by '+str(round(corr_mat[x[j],y[j]],2))+' Spearman rank correlation'
                    else:
                        Summ += '. The categories \''+str(ylabels[x[j]])+"\' and \'"+str(ylabels[y[j]])+'\' are negatively correlated by '+str(round(corr_mat[x[j],y[j]],2))+' Spearman rank correlation'
                    pos = np.count_nonzero((data[:,x[j]]-data[:,y[j]])>0)
                    neg = np.count_nonzero((data[:,x[j]]-data[:,y[j]])<0)
                    if pos<neg and pos == 1:
                        k = np.nonzero((data[:,x[j]]-data[:,y[j]])>0)[0][0]
                        Summ += '. All except for '+str(xlabels[k])+' \''+str(ylabels[y[j]])+'\' is greater than \''+str(ylabels[x[j]])+'\''
                    elif neg<pos and neg == 1:
                        k = np.nonzero((data[:,x[j]]-data[:,y[j]])<0)[0][0]
                        Summ += '. All except for '+str(xlabels[k])+' \''+str(ylabels[y[j]])+'\' is lesser than \'\''+str(ylabels[x[j]])+'\''
                    elif(np.count_nonzero((data[:,x[j]]-data[:,y[j]])<0) == 1):
                        k = np.nonzero((data[:,x[j]]-data[:,y[j]])==0)[0][0]
                        Summ += '. All except for '+str(xlabels[k])+' \''+str(ylabels[y[j]])+'\' is equal to \''+str(ylabels[x[j]])+'\''
    text_file = open(path+"Summary_"+image_name+".txt", "w")
    n = text_file.write(Summ)
    text_file.close()
