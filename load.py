import requests
import pickle
import numpy as np
import yfinance as yf
import pandas as pd
import multiprocessing as mp
from functools import partial
import logging
import datetime
from datetime import date
import time
import vapeplot
from nsetools import Nse
#import nsepy
from fake_headers import Headers
import matplotlib
matplotlib.use('Agg')
from matplotlib import patheffects
from matplotlib import rcParams
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import fuzzymatcher
import mplcyberpunk
import requests_cache
import os
import math
from collections import defaultdict
from folio import *
from breakout import *
import matplotlib.dates as mpl_dates
#from matplotlib import font_manager

#font_dirs = ['.']
#font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

#for font_file in font_files:
#    font_manager.fontManager.addfont(font_file)

logger=logging.getLogger(__name__)

headerr = Headers(headers=False)
nse = Nse()
nsetickers = pd.DataFrame(nse.get_stock_codes().items(), columns=['Symbol', 'Company'])


STONKURLS= ['https://money.rediff.com/companies/most-traded','https://money.rediff.com/gainers','https://money.rediff.com/losers']
NCORES = mp.cpu_count()
FOLIO_COLS = ['Symbol','Buy Price','Returns','Volatility',
              'Skew','Kurtosis','Price','DEMA20', 'DEMA50',
            'RSI','DEMARet','DEMASig','DEMADate','MRet','MSig','MDate','Breakout']

STOP_LOSS = 0.5

if not os.path.exists(os.path.join(os.getcwd(),"data")):
    logger.warning('folder not found')
    os.makedirs(os.path.join(os.getcwd(),"data"))
    os.makedirs(os.path.join(os.getcwd(),"plots"))

    print('folders created')


#CHECK FOR WATCH LISTS

try:
    with open('data\\watchlist.pickle','rb') as file:
        WATCHMAP, BUYDAT = pickle.load(file)
except Exception as e:
    logger.info("init new empty watchlist")
    #WATCHLIST = []
    #WATCHDAT = []
    BUYDAT = []
    WATCHMAP = defaultdict(tuple)
    #list of tuples tickr, buy price, timestamp, current price, delta
    
    #TODO proper folio 

try:
    with open('data\\shoonya_params.txt','r') as file:
        SHLIST = file.read().split(',')
except Exception as e:
    logger.info("no login created")
    SHLIST = ''


try:
    SESSION = requests_cache.CachedSession('yfinance.cache')
except Exception as e:
    logger.warning(e)
    logger.info('Creating request cache')
    requests_cache.install_cache('yfinance.cache')
    SESSION = requests_cache.CachedSession('yfinance.cache')



def scrapetickers(link):
    #scrapes tickers from link, matches against nse tickers
    dfs = pd.read_html(requests.get(link,headers=headerr.generate()).text)
    df = dfs[0]
    if len(df) > 300:
        df = df.head(300)
    matches = fuzzymatcher.fuzzy_left_join(df, nsetickers, left_on = "Company", right_on = "Company")
    return matches


def dema_calc(ticker, period='3mo',interval='1d',lospan=20,hispan=50):
    #UPDATES DAILY DEMA AND SMA CALCS
    #https://randerson112358.medium.com/stock-trading-strategy-using-dema-python-d36e66510a60
    #https://www.roelpeters.be/many-ways-to-calculate-the-rsi-in-python-pandas/
    SESSION.headers = headerr.generate()
    try:            
        df = yf.download(ticker,period=period,interval=interval,session=SESSION)
        df['MA_20'] = df['Adj Close'].ewm(span = lospan, adjust = False).mean()
        df['MA_50'] = df['Adj Close'].ewm(span = hispan, adjust = False).mean()
        df['STD_20'] = df['Adj Close'].rolling(window=lospan).std()
        df['Upper BB'] = df['MA_20'] + (df['STD_20'] * 2)
        df['Lower BB'] = df['MA_20'] - (df['STD_20'] * 2)
        
        df['EMA_20'] = df['Adj Close'].ewm(span = lospan, adjust = False).mean()
        df['EMA_50'] = df['Adj Close'].ewm(span = hispan, adjust = False).mean()
        df['DEMA_20'] = 2*df['EMA_20'] - df['EMA_20'].ewm(span = lospan, adjust = False).mean()
        df['DEMA_50'] = 2*df['EMA_50'] - df['EMA_50'].ewm(span = hispan, adjust = False).mean()
        
        df['up'] = df['Adj Close'].diff().clip(lower=0)
        df['down'] = -1*df['Adj Close'].diff().clip(upper=0)
        df['ema_up'] = df['up'].ewm(com = 13, adjust = False).mean()
        df['ema_down'] = df['down'].ewm(com = 13 , adjust = False).mean()
        df['RSI'] = 100 - (100/(1 + (df['ema_up']/df['ema_down'])))
        df['logR'] = np.log(df['Adj Close']/df['Adj Close'].shift(1))#df['Adj Close'].apply(np.log).diff(1)
        
        return df
    
    except Exception as e:
        logger.debug(f'---->Error downloading {ticker} {e}') 
        return 0

def demadf(df):
    df['Price'],df['DEMASig'],df['DEMADate'],
    df['DEMARet'],df['DEMA20'],
    df['DEMA50'],df['RSI']=zip(*df['Symbol'].apply(dema_ret_calc))                                                              
    return df


            

def statcalc(data, ticker, lospan=20,hispan=50):
    #UPDATES DAILY DEMA AND SMA CALCS
    #https://randerson112358.medium.com/stock-trading-strategy-using-dema-python-d36e66510a60
    #https://www.roelpeters.be/many-ways-to-calculate-the-rsi-in-python-pandas/
    #SESSION.headers = headerr.generate()
    try:
        breakout = False
        df = data[ticker].copy()
        #end = datetime.datetime.now()
        #start =  datetime.datetime.now() - datetime.timedelta(days=period)
        #df = nsepy.get_history(ticker,start=start,end=end)
        #print(df.columns)
    except Exception as e:
        print(f'---->Error looking up {ticker} \n{e}')
        return -99, -99,-99,-99, -99, -99, -99, -99, -99, -99, -99
    else:
        levels, markers = suprezlist(df)
        if has_breakout(levels[-5:],df.iloc[-2],df.iloc[-1]):
            breakout = True
        df['MA_20'] = df['Close'].ewm(span = lospan, adjust = False).mean()
        df['MA_50'] = df['Close'].ewm(span = hispan, adjust = False).mean()
        df['STD_20'] = df['Close'].rolling(window=lospan).std()
        df['Upper BB'] = df['MA_20'] + (df['STD_20'] * 2)
        df['Lower BB'] = df['MA_20'] - (df['STD_20'] * 2)
        df['EMA_20'] = df['Close'].ewm(span = lospan, adjust = False).mean()
        df['EMA_50'] = df['Close'].ewm(span = hispan, adjust = False).mean()
        df['DEMA_20'] = 2*df['EMA_20'] - df['EMA_20'].ewm(span = lospan, adjust = False).mean()
        df['DEMA_50'] = 2*df['EMA_50'] - df['EMA_50'].ewm(span = hispan, adjust = False).mean()
        df['up'] = df['Close'].diff().clip(lower=0)
        df['down'] = -1*df['Close'].diff().clip(upper=0)
        df['ema_up'] = df['up'].ewm(com = 13, adjust = False).mean()
        df['ema_down'] = df['down'].ewm(com = 13 , adjust = False).mean()
        df['RSI'] = 100 - (100/(1 + (df['ema_up']/df['ema_down'])))
        df['logR'] = df['Close'].apply(np.log).diff(1)
        df['Flag'] = np.where((df['DEMA_20'] >= df['DEMA_50']), 1, np.where(df['DEMA_20'] < df['DEMA_50'], -1, np.nan))
        df['Flag'].fillna(method='ffill', inplace=True)
        df['Ret'] = df['Flag'].shift(1)* df['logR']
        df['RetC'] = df['Ret'].cumsum().apply(np.exp)
        
        #momentum strategy
        df['MSig'] = np.sign(df['logR'].rolling(3).mean())
        df['MRet'] = df['MSig'].shift(1) * df['logR']
        
        msignal = df['MSig'].iloc[-1]

        #TODO ? mean reversion strategy

        
        if msignal == 1:
            msignal_date = df[df['MSig']==-1].index.max()# + datetime.timedelta(days=1)
        else:
            msignal_date = df[df['MSig']==1].index.max()# + datetime.timedelta(days=1)
        
        signal = df['Flag'].iloc[-1]
        
        if signal == 1:
            signal_date = df[df['Flag']==-1].index.max()# + datetime.timedelta(days=1)
        else:
            signal_date = df[df['Flag']==1].index.max()# + datetime.timedelta(days=1)

        if lospan ==20 :
            returnz = (np.exp(df['Ret'].mean() * 252)-1) * 100
            mret = (np.exp(df['MRet'].mean() * 252)-1) * 100
        else:
            returnz = (np.exp(df['Ret'].mean() * 75)-1) * 100  #75 5min periods in a day
            mret = (np.exp(df['Ret'].mean() * 75)-1) * 100
        price = df['Close'].iloc[-1]
        
        dema_20 = df['DEMA_20'][-1]
        dema_50 = df['DEMA_50'][-1]
        rsi_now = df['RSI'][-1]


        

        return price, signal, signal_date,returnz, dema_20, dema_50, rsi_now, breakout, msignal, msignal_date, mret  
        #return price, breakout

def dema_ret_calc(ticker,period='1y',interval='1d',lospan=20,hispan=50,plotz=False,alertz=False):
    breakout = False
    try:
        df = dema_calc(ticker,period=period,interval=interval,lospan=lospan,hispan=hispan)

        #print("XXX",breakout)
        #dema strategy
        df['Flag'] = np.where((df['DEMA_20'] >= df['DEMA_50']), 1, np.where(df['DEMA_20'] < df['DEMA_50'], -1, np.nan))
        df['Flag'].fillna(method='ffill', inplace=True)
        df['Ret'] = df['Flag'].shift(1)* df['logR']
        df['Return%'] = df['Ret'].cumsum().apply(np.exp)
        
        #momentum strategy
        df['MSig'] = np.sign(df['logR'].rolling(3).mean())
        df['MRet'] = df['MSig'].shift(1) * df['logR']
        
        msignal = df['MSig'][-1]
        if msignal == 1:
            msignal_date = df[df['MSig']==-1].index.max() #+ datetime.timedelta(days=1)
        else:
            msignal_date = df[df['MSig']==1].index.max() #+ datetime.timedelta(days=1)
        
        signal = df['Flag'][-1]
        
        if signal == 1:
            signal_date = df[df['Flag']==-1].index.max() #+ datetime.timedelta(days=1)
        else:
            signal_date = df[df['Flag']==1].index.max() #+ datetime.timedelta(days=1)


        price = df['Adj Close'][-1]
        dema_20 = df['DEMA_20'][-1]
        dema_50 = df['DEMA_50'][-1]
        rsi_now = df['RSI'][-1]

        
        if lospan ==20 :
            returnz = (np.exp(df['Ret'].mean() * 252)-1) * 100
            mret = (np.exp(df['MRet'].mean() * 252)-1) * 100
        else:
            df = df[df['Close'] != 0]
            returnz = (np.exp(df['Ret'].mean() * 75)-1) * 100  #75 5min periods in a day
            mret = (np.exp(df['Ret'].mean() * 75)-1) * 100

        #df['Date'] = pd.to_datetime(df.index)
        #df['Date'] = df['Date'].apply(mpl_dates.date2num)
        levels, markers = suprezlist(df)
        if has_breakout(levels[-5:],df.iloc[-2],df.iloc[-1]):
            breakout = True
        
    except Exception as e:
        
        if alertz:
            return -99,-99,-99,-99,-99
        
        if plotz:
            return -99,-99,-99,-99,-99,-99,-99,-99, -99, -99, -99, -99

        else:
            return -99,-99,-99,-99,-99,-99,-99,-99, -99, -99, -99
    else:
        if alertz:
            return df,signal,msignal,breakout,price
        if plotz:
            fig = plotbhs(ticker, df, returnz, mret, breakout, levels, markers,lospan)
            return fig, price, signal, signal_date,returnz, dema_20, dema_50, rsi_now, breakout, msignal, msignal_date, mret
        else:
            return price, signal, signal_date,returnz, dema_20, dema_50, rsi_now, breakout, msignal, msignal_date, mret  




def plotbhs(ticker, data, profitz, mret, breakout, levels, markers,lospan):
#def DEMA_plot(ticker, data, returnz, accuracy, levels, breakout):
    '''
    TODO - plot support and rez
    Now plots the forecast price too, funz
    https://www.statology.org/matplotlib-python-candlestick-chart/

    '''
    ticker = ticker.upper()
    plt.xkcd()
    plt.rcParams["font.family"] = "OCR A Std"
    plt.rcParams["figure.autolayout"] = True
    plt.style.use('dark_background')
    vapeplot.set_palette('jazzcup')
    
    plt.rcParams['path.effects'] = [patheffects.withStroke(linewidth=0)]
    #fig, (ax1,ax2,ax3) = plt.subplots(3,sharex=True,figsize=(11,9))
    fig, (ax1,ax3) = plt.subplots(2,sharex=True,figsize=(11,9))
    #ax1.sharex(ax2)
    #ax1.sharex(ax3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    #ax2.spines['top'].set_visible(False)
    #ax2.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax1.plot( data['Adj Close'],  label='Close Price', alpha = 0.95, linewidth = 1.1, color='lime')#plt.plot( X-Axis , Y-Axis, line_width, alpha_for_blending,  label)
    ax1.plot( data['DEMA_20'],  label='DEMA_fast' , alpha = 0.75, linewidth = 1, linestyle='dashed') #plot the Short Term DEMA
    ax1.plot( data['DEMA_50'],  label='DEMA_slo', alpha = 0.75, linewidth = 1,linestyle='dashed') #plot the Long Term DEMA
    ax1.plot( data['Upper BB'], label = 'Upper BB', alpha=0.55, linewidth = 0.75,linestyle='dashdot', color='aqua')
    ax1.plot( data['Lower BB'], label = 'Lower BB', alpha=0.55, linewidth = 0.75,linestyle='dashdot', color='magenta')

    annotate_stuff = f'''PRICE:{round(data['Close'][-1],1)} \nBREAKOUT : {breakout}\n
RSI:{round(data['RSI'][-1])}\n \n{lospan} MA:{round(data['MA_20'][-1],1)}\n'''
        
    if data['Close'][-1] > data['MA_20'][-1]:
        arrow_color = 'green'
        trendz = 'NUMBER UP'
 

    elif data['Close'][-1] < data['MA_20'][-1]:
        arrow_color = 'red'
        trendz = 'DOWN'


    else:
        arrow_color = 'white'
        trendz = 'HODL'

    ax1.annotate(text = f"{annotate_stuff}",fontsize='xx-small',
                 xy = (data.index[-3], data['Close'][-1]),
                 arrowprops = dict(facecolor = arrow_color,shrink=0.5 ))    


    
    #ax2.bar( data.index, data['Volume'], label='Volume')
    #ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
    #ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%d-%m'))
    #plot_all(levels,data,ax3)
    
    width = .25
    width2 = .03

    if lospan != 20:
        width = .0005
        width2 = 0.0002
#define up and down prices
    up = data[data['Close']>=data['Open']]
    down = data[data['Close']<data['Open']]

#define colors to use
    col1 = 'limegreen'
    col2 = 'red'

#plot up prices
    ax3.bar(up.index,up.Close-up.Open,width,bottom=up.Open,color=col1)
    ax3.bar(up.index,up.High-up.Close,width2,bottom=up.Close,color=col1)
    ax3.bar(up.index,up.Low-up.Open,width2,bottom=up.Open,color=col1)

#plot down prices
    ax3.bar(down.index,down.Close-down.Open,width,bottom=down.Open,color=col2)
    ax3.bar(down.index,down.High-down.Open,width2,bottom=down.Open,color=col2)
    ax3.bar(down.index,down.Low-down.Close,width2,bottom=down.Close,color=col2)

    colore = ''
    sup_patch = mpatches.Patch(color='aqua', label='Support')
    rez_patch = mpatches.Patch(color='magenta', label='Resistance')
    for idx, level in enumerate(levels):
        if markers[idx]== 0:

            colore = 'aqua'
        else:

            colore = 'magenta'
        ax3.hlines(level[1], xmin = data.index[0],
                   xmax = max(data.index), linestyle='--',linewidth=0.45,color=colore)

    #rotate x-axis tick labels
    plt.xticks(rotation=65, ha='right',fontsize='xx-small')
    
    
    

    ax1.set_ylabel('Close Price (INR)',fontsize='xx-small')
    
    #ax2.set_ylabel('Volume',fontsize='xx-small')
    #ax2.legend( loc='best', fontsize='x-small')
    ax3.legend( handles=[sup_patch, rez_patch], loc = 'best', fontsize='xx-small')
    ax3.set_ylabel('Prices (INR)',fontsize='xx-small')
    ax1.legend( loc='best', fontsize='xx-small',ncol=2)
    plt.grid(color='w', linestyle='--', linewidth=0.05)
    brk = ''
    if lospan == 20:
        period = '1-YR'
        plt.xlabel('Date-Month',fontsize='xx-small')
    else:
        period = '1 DAY'
        plt.xlabel('5 min',fontsize='xx-small')
    if breakout:
        brk = 'BREAKOUT'
    fig.suptitle(f'{ticker}: {period} DRET% {round(profitz,2)} MRET%\n {trendz} - {brk} ',fontsize='small')
    plt.tight_layout()
    mplcyberpunk.make_lines_glow(ax1)
    mplcyberpunk.make_lines_glow(ax3)

    plt.savefig(f'plots\\{ticker}_DEMA_plot_{date.today()}.png')
    #plt.show()
    plt.close('all')
    return f'plots\\{ticker}_DEMA_plot_{date.today()}.png'





'''
        
def dailydf(choice='100'):
    try:   
        with open(f"data\\{date.today()}_basedf_{choice}.pickle",'rb') as file:
            df = pickle.load(file)
    except Exception as e:
        print(e)
        print(f'Building todays picks for {choice} stonks')
        pool = mp.Pool()
        matches = pool.map_async(scrapetickers, redifflinks)

        if choice == '50':
            df1 = matches.get()[0]["Symbol"].dropna().to_frame()
            df2 = matches.get()[1]["Symbol"].dropna().to_frame()
            pool.close()
            pool.join()

            df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        if choice == '100':
            df1 = matches.get()[0]["Symbol"].dropna().to_frame()
            df2 = matches.get()[1]["Symbol"].dropna().to_frame().head(50)
            df3 = matches.get()[2]["Symbol"].dropna().to_frame().head(50)
            pool.close()
            pool.join()

            df = pd.concat([df1, df2,df3]).drop_duplicates().reset_index(drop=True)
        if choice == '300':
            df1 = matches.get()[0]["Symbol"].dropna().to_frame()
            df2 = matches.get()[1]["Symbol"].dropna().to_frame().head(100)
            df3 = matches.get()[2]["Symbol"].dropna().to_frame().head(100)
            pool.close()
            pool.join()


            df = pd.concat([df1, df2,df3]).drop_duplicates().reset_index(drop=True)
        if choice == '500':
            df1 = matches.get()[0]["Symbol"].dropna().to_frame()
            df2 = matches.get()[1]["Symbol"].dropna().to_frame().head(200)
            df3 = matches.get()[2]["Symbol"].dropna().to_frame().head(200)
            pool.close()
            pool.join()


            df = pd.concat([df1, df2,df3]).drop_duplicates().reset_index(drop=True)
        df['Symbol'] = df['Symbol'] + '.NS'
        df[['Price','DEMA20', 'DEMA50',
            'RSI','DEMARet','DEMASig','DEMADate','Breakout',
            'MSig','MDate','MRet']] = ['']*11
        #price, signal, signal_date, returnz, dema_20, dema_50, rsi_now
        #breakout, mom sig, mom date, mom ret
        data = yf.download(tickers=list(df['Symbol']),group_by='ticker')
        partstat = partial(statcalc,data)
        df['Price'],df['DEMASig'],\
                                    df['DEMADate'],df['DEMARet'],\
                                    df['DEMA20'],df['DEMA50'],\
                                    df['RSI'], df['Breakout'],\
                                    df['MSig'],df['MDate'],df['MRet']=zip(*df['Symbol'].apply(partstat))
        
        with open(f"data\\{date.today()}_basedf_{choice}.pickle",'wb') as file:
            pickle.dump(df,file)

    
    return df


def dailydf_opt(df,choice,rupees=1000,numbaz=1000,rf=0.01):
    #get opt risk folio, and update by weights
    
    #try:
        #with open(f"data\\dayfolio_{date.today()}_{choice}.pickle",'rb') as file:
            #f_df, assets, folios, figz = pickle.load(file)
    #except Exception as e:
        #print(e)
    if len(df[df['Breakout']==True]) == 0:
        print('No breakout found, choosing by DEMA crossover')
        df = df[df['DEMASig'] == 1]
    else:
        df = df[df['Breakout'] == True]
    f_df, assets, folios, figz = opt_folio_ratio(df,rupees,numbaz,rf)
        #df = df[df[datetime.datetime.now() - df['DEMADate'] < pd.Timedelta(15,unit='d')]]
        #corr_df = get_corr_df(df)
        #corrfig = plot_corr_df(corr_df)
        #with open(f"data\\dayfolio_{date.today()}_{choice}.pickle",'wb') as file:
            #pickle.dump((f_df, assets, folios, figz), file)

    return f_df, assets, folios, figz



'''


#if __name__ == "__main__":
    #a = dema_ret_calc('DOGE-INR', plotz=True)
    #def dema_ret_calc(ticker,period='1y',interval='1d',df_ret=False,plotz=False):
    #pass
'''
    ticker_list = ['VIKASLIFE.NS','RELIANCE.NS']

    df = pd.DataFrame()
    df['Symbol'] = ticker_list
    data = yf.download(
     
    # passes the ticker
    tickers=ticker_list,
     
    # used for access data[ticker]
    group_by='ticker',
 
    )
    partstat = functools.partial(statcalc,data)
    df['Price'],df['Breakout']=zip(*df['Symbol'].apply(partstat))
    print(df)

    print('CHUNKED MP TIME')
    choice = '50'
    start = time.time()
    df = chunkdailydf(choice)
    #foliorez = dailydf_opt(df,choice)
    print('TIME CHUNKED-->',time.time() - start, "seconds")
    print('DELETING FILES')
    os.remove(f"data\\{date.today()}_basedf_{choice}.pickle")
    #os.remove(f"data\\dayfolio_{date.today()}_{choice}.pickle")
    start = time.time()
    print('STRAIGHT APPLY')
    df = dailydf(choice)
    print('TIME APPLY-->',time.time() - start, "seconds")
    
'''
