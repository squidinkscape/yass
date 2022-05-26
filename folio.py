import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vapeplot
import pickle
import yfinance as yf
import datetime
import functools
import logging
from datetime import date
import mplcyberpunk
import math
import random
from matplotlib import patheffects
from matplotlib import rcParams
import matplotlib.cm as cm

logger = logging.getLogger(__name__)


def coskew(df):
    
    # Number of stocks
    num = len(df.columns)
    
    # Two dimionsal matrix for tensor product 
    mtx = np.zeros(shape = (len(df), num**2))
    
    v = df.values
    means = v.mean(0,keepdims=True)
    v1 = (v-means).T
    
    for i in range(num):
        for j in range(num):
                vals = v1[i]*v1[j]
                mtx[:,(i*num)+j] = vals/float((len(df)-1)*df.iloc[:,i].std()*df.iloc[:,j].std())
    
    #coskewness matrix
    m3 = np.dot(v1,mtx)
    
    #Normalize by dividing by standard deviation
    for i in range(num**2):
        use = i%num
        m3[:,i] = m3[:,i]/float(df.iloc[:,use].std())
    
    return m3

def cokurt(df):
    # Number of stocks
    num = len(df.columns)
    
    #First Tensor Product Matrix
    mtx1 = np.zeros(shape = (len(df), num**2))
    
    #Second Tensor Product Matrix
    mtx2 = np.zeros(shape = (len(df), num**3))
    
    v = df.values
    means = v.mean(0,keepdims=True)
    v1 = (v-means).T

    for k in range(num):
        for i in range(num):
            for j in range(num):
                    vals = v1[i]*v1[j]*v1[k]
                    mtx2[:,(k*(num**2))+(i*num)+j] = vals/float((len(df)-1)*df.iloc[:,i].std()*\
                                                                df.iloc[:,j].std()*df.iloc[:,k].std())

    m4 = np.dot(v1,mtx2)
    for i in range(num**3):
        use = i%num
        m4[:,i] = m4[:,i]/float(df.iloc[:,use].std())
        
    return m4




#@functools.lru_cache(maxsize=16) #? ? ???
def opt_folio_ratio(buy_df,rupees=10000,numbaz=3000, rf=0.02):
    '''
    buy-df- df of potential buys (ticker in Symbol)
    rupeez - rupeez to blow up on folio
    numbaz - no. of folio opts.
    rf - risk factor (5 % - 0.05)
    https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
    '''

    f_df = pd.DataFrame() #init empty folio df

    #get table df
    
    for ticker in buy_df['Symbol']:
        try:
            df = yf.download(ticker,period='5y',interval='1d')
            
        except Exception as e:
            print(str(e))
            continue
        else:   
            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
            try:
                
                if f_df.empty:
                    f_df = df
                else:
                    f_df = f_df.join(df, how='outer')
            except Exception as e:
                print(e)
                continue

        

    
    log_ret = np.log(f_df/f_df.shift(1)).dropna()

    cov_df = f_df.pct_change().apply(lambda x: np.log(1+x)).cov()
    
    corr_df = f_df.pct_change().apply(lambda x: np.log(1+x)).corr()

    
    ind_er = f_df.resample('Y').last().pct_change().mean()

    ann_sd = f_df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))


    #vapeplot smokes
    vapeplot.set_palette('cool')
    plt.style.use('dark_background')
    
    skew_arr, kurt_arr = coskew(log_ret), cokurt(log_ret)

    assets = pd.concat([ind_er, ann_sd], axis=1)
    assets.columns = ['Returns', 'Volatility']
        
    p_ret, p_vol, p_weights, p_skew, p_kurt = [],[],[],[],[]
    
    #CHUNK THIS
    for portfolio in range (numbaz):
        weights = np.random.random(len(f_df.columns))
        weights = weights/np.sum(weights)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er)
        p_ret.append(returns)
        var = cov_df.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        sd = np.sqrt(var)
        ann_sd = sd*np.sqrt(250)
        p_vol.append(ann_sd)
        skew = np.dot(weights.T, np.dot(skew_arr,np.kron(weights,weights)))
        p_skew.append(skew)
        kurt = np.dot(weights.T, np.dot(kurt_arr,np.kron(weights,np.kron(weights,weights))))
        p_kurt.append(kurt)
        
    data = {'Returns':p_ret, 'Volatility':p_vol, 'Skew':p_skew, 'Kurtosis':p_kurt}

    for indx, symbol in enumerate(f_df.columns):
        
        try:
            
            data[symbol+'_weight'] = [w[indx] for w in p_weights]
            price = buy_df[buy_df['Symbol']==symbol]['Price']
            data[symbol+'_quantity'] = [int(math.floor(w[indx]*rupees/price)) for w in p_weights]
        
        except Exception as e:
            #do nothing
            continue
    folios = pd.DataFrame(data)

        
    #print(f'----folios saved HAH----\n')
    
    min_vol_folio = folios.loc[folios['Volatility'].idxmin()]
    #max_ret_folio = folios.loc[folios['Returns'].idxmax()] ? ?  ?
    folios['SR']  = pd.to_numeric((folios['Returns']-rf)/folios['Volatility'], errors='ignore')
    opt_risk_folio = folios.iloc[folios['SR'].idxmax()]
    #opt_risk_folio = folios.iloc[((folios['Returns']-rf)/folios['Volatility']).idxmax()]
    
    plt.xkcd()
    plt.rcParams["font.family"] = "OCR A Std"
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['path.effects'] = [patheffects.withStroke(linewidth=0)]
    plt.rcParams['figure.figsize'] = (9,7)
    plt.style.use('dark_background')
    vapeplot.set_palette('avanti')
    #colors = np.random.randint(low = 1, high = 3, size = len(folios))#np.linspace(0, 1, len(folios))
    #colormap = np.array(['r', 'g', 'b'])
    colors = cm.rainbow(np.linspace(0, 1, len(folios)))
    #plt.legend()
    plt.scatter(folios['Volatility'],folios['Returns'], marker='.', c = colors,s=10, alpha=0.4)   
    plt.scatter(min_vol_folio[1], min_vol_folio[0], color='b',marker='o', label='MinVlt', s=100)
    #plt.scatter(max_ret_folio[1], max_ret_folio[0], color='r',marker='^', label='MaxRet',s=100)
    plt.scatter(opt_risk_folio[1], opt_risk_folio[0], color='g',marker='D', label='OptRsk',s=130)
    plt.legend(loc='best')
    plt.xlabel('volatility')
    plt.ylabel('returns')
    plt.title('efficient frontier')
    mplcyberpunk.make_scatter_glow()
    plt.savefig(f"plots\\opt_folios_{date.today()}.png")
    
    figz = f"plots\\opt_folios_{date.today()}.png"
    
    
    return f_df, assets, folios, figz


def price_yf(ticker):
    try:
        
        df = yf.download(ticker)   
        return df['Adj Close'].iloc[-1]
    except Exception as e:
        return -99
    
def folio_initfromfile(filename):
    #returns f_df,assets, folios,figz
    try:
        with open(filename) as file:
            df = pd.read_csv(file)
    except Exception as e:
        print(e)
        logger.warning("ERRORE, errore - Failed opening csv file")
        return None
    
    print('read folio from disk')
    if len(df) == 0:
        print("No stonks in folio")
    
        #df['Symbol'] = df['Symbol'] + ".NS"
        #if df['Buy Date'].empty:
    df['Price'] = []
    dropz = []
        #df['Buy Date'] = pd.to_datetime(df['Buy Date'], infer_datetime_format=True)
    for index, row in df.iterrows():
        try:
            
            row['Symbol'] = row['Symbol'] + ".NS"
            row['Price'] = row['Symbol'].apply(price_yf)
            if row['Price'] == -99 : dropz.append(index)
        except Exception as e:
            print(e)
            print('Bad ticker')
            dropz.append(index)
            
    df.drop(dropz, inplace=True)
    cols = ['Number', 'Buy Price']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)
        #df['SimpleR'] = (df['Number']*df['Price'] - df['Number']*df['Buy Price'])*(1/(df['Number']*df['Buy Price'])) *(datetime.datetime.now()-df['Buy Date']).dt.days
        #df['LogR'] = np.log(df['SimpleR']+1)
    try:
        
        #corr_df = get_corr_df(df)
        #cfig = plot_corr_df(corr_df)
        f_df, assets, folios, figz = opt_folio_ratio(df,1000,1000,0.01)
        
    except Exception as e:
        logger.info(e)
        return None
    else:
        return df, f_df,assets, folios, figz 
    
def init_folio(filename):
    #reads folio from csv
    try:
        df, f_df,assets, folios,figz  = folio_initfromfile(filename)
        return df, f_df,assets, folios,figz
        #folio, rez = folio_fromfile(filename)
    except Exception as e:
        logger.error(e)
        logger.warning('ERRORE,errore ---> Could not read folio')
        return -99,-99,-99,-99,-99,-99
    

def optlistfolio(liststr,rupees=10000,numbaz=2000,rf=0.01):
    df = pd.DataFrame()
    df['Symbol'] = pd.Series(liststr)
    df['Price'] = df['Symbol'].apply(price_yf)
    #dropz = df[ df['Price'] == -99 ].index
    #df.drop(dropz, inplace = True)

    #corr_df = get_corr_df(df)
    #cfig = plot_corr_df(corr_df)
    f_df, assets, folios, figz = opt_folio_ratio(df,rupees,numbaz,rf)
    #min_vol_folio = folios.loc[folios['Volatility'].idxmin()]
    #max_ret_folio = folios.loc[folios['Volatility'].idxmax()]
    #opt_risk_folio = folios.loc[folios['SR'].idxmax()]
    return df,f_df, assets, folios, figz

