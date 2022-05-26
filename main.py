from load import *
from folio import *
from asciiart import *
import os
import logging
import sys
from tabulate import tabulate
import PySimpleGUI as sg
from textwrap import wrap
import warnings
import threading
import random
import concurrent.futures
import plotext as plx
import matplotlib.font_manager as font_manager


#from shoonya import Shoonya, TransactionType, ProductType, OrderType, InstrumentType
#import matplotlib.font_manager as font_manager
#font_dir = ['/fonts']
#for font in font_manager.findSystemFonts(font_dir):
#    font_manager.fontManager.addfont(font)
#from shoonyapy import ShoonyaApi

ALERTS = False
THREAD_EVENT = '-THREAD-'
cp = sg.cprint

SIGNALZ = {1:'BUY',-1:'SELL',0:'HODL'}
INTERVALZ = {'day':'1d','iday':'15m'}
#PERIODZ = {'day':'6mo','iday':'1mo'}

now = datetime.datetime.now().hour


if 9 <= now and now < 13:
    SESH = 'un'
elif 13 <= now and now <= 16:
    SESH = 'dos'
else:
    SESH = 'tres'


def warn(*args, **kwargs):
    pass

warnings.warn = warn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data\\yass.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger()



def popupimage(title, filename, message, width=70):

    lines = list(map(lambda line:wrap(line, width=width), message.split('\n')))
    height = sum(map(len, lines))
    message = '\n'.join(map('\n'.join, lines))

    layout = [[sg.Titlebar(LOGO)],
        [sg.Image(filename=filename, expand_x=True)],
        [sg.Text(message, size=(width, height), justification='center', expand_x=True)]
    ]

    sg.Window(title, layout, keep_on_top=True,modal=True).read(close=10)
    

def popupdaily():
    
    choices = ('volume', 'gainers', 'losers', 'cancel')

    layout = [  [sg.Radio('day', "Rg", default=True,key='-R1-',enable_events=True),
                 sg.Radio('iday', "Rg", default=False,key='-R2-',enable_events=True)],
                [sg.Text('Click to scrape')],
                [sg.Listbox(choices, size=(15, len(choices)),
                            key='-choice-', enable_events=True)] ]

    windowdaily = sg.Window('daily picks', layout, no_titlebar=True, alpha_channel = 0.95)
    choice = ''
    iday = False
    interval = 'day'
    while True:                  # the event loop
        event, values = windowdaily.read()
        if event == sg.WIN_CLOSED:
            break
        if values['-R1-']:
            iday = False
        if values['-R2-']:
            iday = True
        if values['-choice-']:    # if something is highlighted in the list
            choice = values['-choice-'][0]
            #sg.popup(f"checking {values['-choice-'][0]}")
            break
    windowdaily.close()
    del windowdaily
    if iday:
        interval = 'iday'
    return choice, interval

def popupbhs():
    
    #choices = ('volume', 'gainers', 'losers', 'cancel')

    layout = [  [sg.Radio('day', "Rg", default=True,key='-R1-',enable_events=True),
                 sg.Radio('iday', "Rg", default=False,key='-R2-',enable_events=True)],
                [sg.Text('enter ticker'),sg.Input('',key = '-tickr-')],
                [sg.Button('Cancel')]]

    windowbhs = sg.Window('daily picks', layout, no_titlebar=True, alpha_channel = 0.95,finalize=True)
    ticker = ''
    interval_day = True
    windowbhs['-tickr-'].bind("<Return>", "_Enter")

    while True:                  # the event loop
        event, values = windowbhs.read()
        if event == 'Cancel':
            break
        if values['-R1-']:
            interval_day = True
        if values['-R2-']:
            interval_day = False
        if values['-tickr-']:    
            ticker = values['-tickr-']
            break
    windowbhs.close()
    del windowbhs
    if interval_day:
        interval = 'day'
    else:
        interval = 'iday'
    return ticker, interval

def popupfolio():
    
    choices = list()
    choices.append('new')
    choices.append('list')
    choices.append('cancel')
    layoutfolio = [ [sg.Text('load/enter folio')],
                [sg.Listbox(choices, size=(15, len(choices)),
                            key='-choice-', enable_events=True)] ]

    windowfolio = sg.Window('folio', layoutfolio, no_titlebar=True,
                            alpha_channel = 0.95,modal=True)

    rez = []
    choice = ''
    while True:                 
        eventf, valuesf = windowfolio.read()
        
        if eventf == sg.WIN_CLOSED:
            break
        
        if valuesf['-choice-']:    
            choice = valuesf['-choice-'][0]
            rez.append(choice)
            break

    windowfolio.close()
    del windowfolio
    return choice


def checkforfile(fname):
    if fname is None:
        return False
    if os.path.isfile(fname):
        if '.csv' in fname:
            return True
    return False

def bhsrez(ticker,interval):
    if interval == 'day':
        prd = '6mo'
        itr = '1d'
        lospan = 20
        hispan = 50
    else:
        prd = '1d'
        itr = '5m'
        lospan = 2
        hispan = 10
        
    try:
        fig, price, signal, \
             signal_date,returnz, dema_20, dema_50,\
             rsi_now, breakout, msignal, msignal_date, \
             mret = dema_ret_calc(ticker,period=prd,
                                  interval=itr,lospan=lospan,
                                  hispan=hispan,plotz=True)
        
    except Exception as e:
        return 'INVALID TICKER'
    else:
        if fig == -99:
            return 'INVALID TICKER'
        brk = 'NOPE'
        if breakout:
            brk = 'BREAKOUT!'
        rezstring = f'''{ticker} --> Rs. {price} \n \
    last signal DEMA {signal} on {signal_date} \n \
    M {msignal} on {msignal_date}
    sim DEMA ret {round(returnz,2)} M ret {round(mret,2)} \n \
    rsi today {round(rsi_now,2)} \n Breakout? - {brk}'''
        return fig, rezstring

#def reportdaily(df):
   
    


def dailydfrez(choice,interval):
    df = chunkpickdf(choice,interval)
    
    df.sort_values(by=['RSI','Price'],ascending=True,inplace=True)
    #text = reportdaily(df)
    
    
    today = datetime.datetime.now()

    if interval == 'day':
        demadf = df[(df['DEMASig']==1) & ((today - df['DEMADate']).dt.days < 3)]
        demadf['DEMADate'] = demadf['DEMADate'].dt.strftime("%b %d %y")
        demadf = demadf[['Symbol','Price','DEMA20', 'DEMA50','RSI','DEMARet','DEMADate','DEMASig']]
    else:
        demadf = df[df['DEMASig']==1]
        demadf = demadf[['Symbol','Price','DEMA20', 'DEMA50','RSI','DEMARet','DEMASig']]
        
    demadf.sort_values(by='DEMARet',ascending=False,inplace=True)

    
    
    msigdf = df[['Symbol','Price','MSig', 'MDate','MRet','RSI','DEMASig']]
    if interval == 'day':
        msigdf = msigdf[(msigdf['MSig']==1) & ((today - df['MDate']).dt.days < 3)]
        msigdf['MDate'] = msigdf['MDate'].dt.strftime("%b %d %y")
    else:
        msigdf = msigdf[msigdf['MSig']==1]
        msigdf = msigdf[['Symbol','Price','MSig','MRet','RSI','DEMASig']]
    msigdf.sort_values(by='MRet',ascending=False,inplace=True)
    

    dfpicks = df[df['Breakout']==True]
    if interval == 'day':
        dfpicks = dfpicks[['Symbol','Price','DEMA20', 'DEMA50','RSI','DEMARet','DEMASig','DEMADate','MSig','Breakout']]
    else:
        dfpicks = dfpicks[['Symbol','Price','DEMA20', 'DEMA50','RSI','DEMARet','DEMASig','MSig','Breakout']]
    text = '\n---DEMA signals---\n'
    text += text2art('DEMA',font='random') + '\n'
    text += tabulate(demadf, headers='keys', tablefmt='psql',floatfmt=".1f",showindex=False)
    text += '\n---Momentum signals---\n'
    text += text2art('MOMNT',font='random') + '\n'
    text += tabulate(msigdf, headers='keys', tablefmt='psql',floatfmt=".1f",showindex=False)
    text += '\n---Price action signals---\n'
    text += text2art('BRK',font='random') + '\n'
    text += tabulate(dfpicks, headers='keys', tablefmt='psql',floatfmt=".1f",showindex=False)

    return text, df

def alertthread(window):
    
    i = 0
    while True:
        if not ALERTS:
            break
        with open('data\\watchlist.pickle','rb') as file:
            WATCHMAP, BUYDAT = pickle.load(file)
            
        if len(WATCHMAP) == 0:
            cp('no stonks found in watchmap, exiting')
            break
            
        for ticker,val in WATCHMAP.items():

            buy_price = WATCHMAP[ticker][2]
            SESSION.headers = headerr.generate()
            txt = '\n' + text2art(f'{ticker}',font='random') + '\n'
            
            df,signal,msignal,breakout,price = dema_ret_calc(ticker,period='1d',interval='5m',plotz=False,alertz=True,lospan=2,hispan=10)

            rez = (price - buy_price)/buy_price
            delta = round(100*rez,2)
            cflag = ''
            dsig = 'hodl'
            msig = 'hodl'
            if signal == 1:
                dsig = 'buy'
            elif signal == -1:
                dsig = 'sell'
            if msignal == 1:
                msig = 'buy'
            elif msignal == -1:
                msig = 'sell'
                
            #dsig = SIGNALZ[int(signal)]
            #msig = SIGNALZ[int(msignal)]
            if delta > 0 and signal == 1:
                flag = f'+ D-{dsig},M-{msig}'
                cflag = 'blue on black'
            if signal == -1 or msignal == -1:
                flag = f'+ D-{dsig},M-{msig}'
                cflag = 'yellow on black'
            if delta < 0 - STOP_LOSS:
                flag = 'REKT'
                WATCHMAP[ticker][0] == flag
                cflag = 'red on black'
            else:
                flag = f'- D-{dsig},M-{msig}'
                cflag = 'white on black'

            txt += tabulate(df.iloc[:,:5].tail(5), headers='keys', tablefmt='psql')
            txt += '\n\n\n\n'
            

            x = range(len(df))
            y = df['Adj Close']
            #y0 = df['Open']
            y1 = df['DEMA_20']
            y2 = df['DEMA_50']
            
            plx.clf()
            plx.clear_color()

            plx.plot_size(75,35)
            plx.plot(x,y,label='close')
            plx.scatter(x,y1,label='fast_dema',marker='x')
            plx.scatter(x,y2,label='slow_dema',marker='-')
            plx.title(f'tickr:{ticker} now Rs:{round(price,2)} DELTA:{delta}\n)')
            rez = plx.build()
            r = plx.uncolorize(rez)
            
            plx.clf()
            txt += r

            
            txt += f'TICKER: {ticker}\nPRICE : {round(price,2)}\nDELTA: {delta}\nDEMA SIG- {dsig}\nMOMENTUM SIG - {msig}\n'
            
            window.write_event_value('-THREAD-', f'{ticker} --> DONE')
            
            cp(txt,c=cflag)
            time.sleep(3)
    
        time.sleep(300)
        
    

def scrapedailydf(choice='volume',yfinance=True):
    
    pool = mp.Pool(NCORES)
    matches = pool.map_async(scrapetickers, redifflinks)

    if choice == 'volume':
        df1 = matches.get()[0]["Symbol"].dropna().to_frame()
        df2 = matches.get()[1]["Symbol"].dropna().to_frame().head(100)
        df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        pool.close()
        pool.join()
    if choice == 'gainers':
        
        df = matches.get()[1]["Symbol"].dropna().to_frame().head(200).drop_duplicates().reset_index(drop=True)
        pool.close()
        pool.join()

    if choice == 'losers':
        df = matches.get()[2]["Symbol"].dropna().to_frame().head(200).drop_duplicates().reset_index(drop=True)
        #df2 = matches.get()[1]["Symbol"].dropna().to_frame().head(150)
        #df3 = matches.get()[2]["Symbol"].dropna().to_frame().head(100)
        pool.close()
        pool.join()
        
        #df = pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)
        
    if yfinance:        
        df['Symbol'] = df['Symbol'] + ".NS"
    print("\n",'SCRAPED ----->\n',text2art(f'{len(df)}',font='random'))

    return df

def chunkhelper(partfunc,df):
    df['Price'],df['DEMASig'],\
                                df['DEMADate'],df['DEMARet'],\
                                df['DEMA20'],df['DEMA50'],\
                                df['RSI'],df['Breakout'],\
                                df['MSig'],df['MDate'],df['MRet']=zip(*df['Symbol'].apply(partfunc))
    return df

def chunkpickdf(choice='volume',interval='day'):

    now = datetime.datetime.now().hour

    if 9 <= now and now < 13:
        SESH = 'un'
    elif 13 <= now and now <= 16:
        SESH = 'dos'
    else:
        SESH = 'tres'
    
    try:
        with open(f"data\\{date.today()}_basedf_{choice}__{interval}_{SESH}.pickle",'rb') as file:
            df = pickle.load(file)
    except Exception as e:
        #print(e)
        logger.info('no base df found')
        df = scrapedailydf(choice,True)
        tickers = list(df['Symbol'])
        df[['Price','DEMA20', 'DEMA50',
            'RSI','DEMARet','DEMASig','DEMADate','Breakout',
            'MSig','MDate','MRet']] = ['']*11
        
        chunks = list()
        dfchunks = list()
        num_chunks = NCORES

        if len(df) < 200:
            num_chunks == 4
        
        #if num_chunks < 4:
        #   print(text2art('SLO',font='random'))
        #   num_chunks == 4
        
        chunk_size = len(df)//num_chunks
        #pool = mp.Pool(NCORES)
        #partchunk = partial(chunkhelper,partstat)

        if interval == 'iday':
            prd = '1d'
            itr = '5m'
            lospan = 2
            hispan = 10
        else:
            prd = 'ytd'
            itr = '1d'
            lospan = 20
            hispan = 50
            
        with concurrent.futures.ProcessPoolExecutor() as executor:
            
            for i in range(num_chunks):
                print(f'process {i} -- >')
                SESSION.headers = headerr.generate()
                if i == num_chunks - 1:
                    #data = nsepythreaded()
                    data = yf.download(tickers[i*chunk_size:],group_by='ticker',period=prd,interval=itr)
                    #chunks.append(df[i*chunk_size:])
                    partstat = partial(statcalc,data,lospan=lospan,hispan=hispan)
                    chunks.append(executor.submit(chunkhelper,partstat,df[i*chunk_size:]))
                    time.sleep(1.5)
                else:
                    data = yf.download(tickers[i*chunk_size:(i+1)*chunk_size],group_by='ticker')
                    #chunks.append(df[i*chunk_size:(i+1)*chunk_size])
                    partstat = partial(statcalc,data,lospan=lospan,hispan=hispan)   
                    #p = mp.Process(target=chunkhelper,args=[partstat,df[i*chunk_size:(i+1)*chunk_size]])
                    chunks.append(executor.submit(chunkhelper,partstat,df[i*chunk_size:(i+1)*chunk_size]))
                    time.sleep(1.5)
                print(f'process {i} spawned and slept')

        cp(text2art('CALC',font='random'),'purple on black')
        
        #for i in range(15): print('>',end='')
        
        for rez in concurrent.futures.as_completed(chunks):
            dfchunks.append(rez.result())
        df = pd.concat(dfchunks)
        with open(f"data\\{date.today()}_basedf_{choice}__{interval}_{SESH}.pickle",'wb') as file:
            pickle.dump(df,file)
        #print('NAFTER----->',len(df))
    return df


  
if __name__ == "__main__":
    mp.freeze_support()
        
    sg.theme('DarkPurple2')
    font = ("Press Start 2P",12)

    sg.set_options(font=font)
    layout = [[sg.Titlebar(LOGO)],
              [sg.Button('picks'),sg.Button('bhs'),
               sg.Button('folio'),sg.Button('alerts'),sg.Button('del')],
        
              [sg.Multiline(MAIN_MENU,size=(400,120), key='-out-', reroute_cprint=True,
                            reroute_stdout=True, autoscroll= True, write_only=True,text_color='green',background_color='black')]]


    window = sg.Window('yass', layout, size=(640,480),auto_size_buttons = True,grab_anywhere=True,finalize=True)
    #window.set_cursor('hand1')

    while True:

        event, values = window.read()
        cp(event, values)
        
        if event == THREAD_EVENT:
            cp(f'watchlist alert', colors='purple on black', end='\n')
            #cp(f'{values[THREAD_EVENT]}')
        
        if event == sg.WINDOW_CLOSED:
            break
        #get picks
        
        if event == 'folio':
            choice = popupfolio()
            if choice == 'cancel':
                print('folio load cancelled')
                
            if choice == 'new':    
                fname = sg.popup_get_text('enter path of .csv file',title=CSVLOGO)
                print(f'{fname}')
                if not checkforfile(fname):
                    print('Cannot find file specified')
                else:
                    window.perform_long_operation(lambda: init_folio(fname), '-folioread-')

            if choice == 'list':
                liststr = sg.popup_get_text('ticker list',title='enter ticker list',
                                            no_titlebar=True)
                if liststr == None or liststr == '':
                    print('Nothing entered')
                
                else:
                    liststr = liststr.upper()
                    liststr = liststr.split(',')
                    tickrs = []
                    for tickr in liststr:
                        try:
                            df = yf.download(tickr, period='7d', interval='1d')
                            tickrs.append(tickr)
                        except:
                            print('Invalid ticker {tickr}')
                            continue
                            
                    print(f'checking optimal ratio for : \n{tickrs}')
                    rupees = sg.popup_get_text('rupees in',title='amount in rs',
                                            no_titlebar=True)
                    window.perform_long_operation(lambda: optlistfolio(tickrs,rupees), '-listfolio-')

                    
        if event == '-listfolio-' or event == '-folioread-':
            try:
                df, f_df, assets, folios, figz = values[event]
            except Exception as e:
                print(e)
                print('ERRORE, errore --> folio opt error')
            else:
                cp(text2art('FOLIOZE',font='random'),c='')
                opt_risk_folio = folios.iloc[((folios['Returns']-0.05)/folios['Volatility']).idxmax()]
                min_vol_folio = folios.loc[folios['Volatility'].idxmin()]
                #max_ret_folio = folios.loc[folios['Returns'].idxmax()]
                print("optimal risk folio:\n")
                cp(text2art('OPT',font='random'),c='')
                cp(opt_risk_folio,c='')
                cp("min. volatility folio:\n",c='white on black')
                cp(min_vol_folio,c='white on black')
                #cp('max return folio:\n',c='red on black')
                #cp(max_ret_folio,c='red on black')
            #print(tabulate(df,headers='keys',tablefmt='psql'))
                popupimage('folio',figz,'efficient frontier')
                
                
        if event == 'bhs':
            ticker, interval = popupbhs()
            #text = sg.popup_get_text('enter ticker:',no_titlebar=True)
            if ticker in (None, 'Cancel','','Ok'):
            #if text == 'Cancel' or text == None or text == '':
                print('Nothing entered')
                continue
            else:
                print(f'checking {interval.upper()} bhs for {ticker}')
                #print(ticker,interval)
                rez = bhsrez(ticker.upper(),interval=interval)
                if rez == 'INVALID TICKER':
                    #window['-out-'].update('invalid ticker')
                    print('invalid ticker')
                else:
                    print(f'{rez[1]}')
                    #window['-out-'].update(f'{rez[1]}')
                    popupimage('rez',rez[0],f'{ticker} {interval}')
 

        if event == 'picks':
            choice, interval = popupdaily()
            if choice == 'cancel':
                print('Pick scrape cancelled')
            else:
                print(f'checking {choice} stonk picks for today\n')

                window.perform_long_operation(lambda: dailydfrez(choice,interval), '-daily rez-')

        if event == '-daily rez-':
            text, df = values[event]
            
            savecsv = sg.popup_ok('save results to .csv?',title=SAVELOGO)

            if savecsv == 'Ok' or savecsv:
                
                df.to_csv(f'data\\dailyrezdf_{choice}.csv')
                print(f'saved data\\dailyrezdf_{choice}.csv')
                
            cp(text,c='')
            sg.popup_scrolled(text, title=DAILYLOGO, size=(40, 40),
                              text_color='blue',background_color='black',non_blocking=True,modal=False)
            
        if event == 'alerts':
            if len(WATCHMAP) == 0:
                print('watchlist empty, enter list in the popup:')
            else:
                print('Enter tickers to add to watchlist')
                    
            liststr = sg.popup_get_text('enter watchlist tickers',title='enter watchlist',
                                                no_titlebar=True)
                    
            if liststr is None or liststr == '':
                print('No ticker entered, try again')
            else:
                liststr = liststr.upper()
                liststr = liststr.split(',')

                    #check for invalid tickers
                #tickrs = []
                price = 0
                for tickr in liststr:
                    if tickr not in WATCHMAP:                            
                        try:
                            df = yf.download(tickr, period='7d', interval='1d')
                        except:
                            print('Invalid ticker {tickr}')
                        else:
                            #df.reset_index()
                            
                            #WATCHLIST.append(tickr)
                            #WATCHDAT.append(df.iloc[-1].to_numpy(dtype='float'))
                            price = df['Adj Close'].iloc[-1]

                            WATCHMAP[tickr] = ('WATCH',df.iloc[-1].to_numpy(dtype='float'), price)
                            
                #WATCHLIST.extend(tickrs)
                print('WATCHMAP-->\n',WATCHMAP)
                print('added to watchlist')
                with open('data\\watchlist.pickle','wb') as file:
                    pickle.dump((WATCHMAP,BUYDAT),file)
            if len(WATCHMAP) != 0 and not ALERTS:
                ALERTS = True
                sg.popup('alerts enabled',no_titlebar=True)
                threading.Thread(target=alertthread, args=(window,), daemon=True).start()

                
        if event == 'del':
            if len(WATCHMAP) == 0:
                print('watchlist empty :p')
            else:
                print('Enter tickers to delete from watchlist')    
                liststr = sg.popup_get_text('delete tickers',title='delete tickers',
                                                no_titlebar=True)
                    
                if liststr is None or liststr == '':
                    print('No ticker entered, try again')
                else:
                    liststr = liststr.upper()
                    liststr = liststr.split(',')

                    for tickr in liststr:
                        if tickr not in WATCHMAP:                            
                            print(f'ticker {tickr} not found in watchlist?')
                            continue
                        else:
                            if tickr not in BUYDAT:
                                WATCHMAP.pop(tickr)
                            else:
                                cp('ticker is in active buy list',c='red on black')
                    print('WATCHLIST-->\n',WATCHMAP)
                    print('removed {liststr} from watchlist')
                    with open('data\\watchlist.pickle','wb') as file:
                        pickle.dump((WATCHMAP,BUYDAT),file)


    window.close()
    del window
    sys.exit()



'''
        if event == 'trade':
            if SHLIST == '': 
                text = sg.popup_get_text('enter user id, password, pan.no:',no_titlebar=True)
                if text in (None, 'Cancel','','Ok'):
                    print('Nothing entered')
                    continue
                else:
                    if len(text.split(',')) != 3:
                        print('Try again - user id, pass, pan.no')
                    else:
                        SHLIST = text.split(',')
                        cp(f'Trying to login with {SHLIST}')
                        window.perform_long_operation(lambda: shoonyatst(SHLIST), '-trade login-')
            else:
                cp(f'Trying to login with {SHLIST}')
                #try:
                window.perform_long_operation(lambda: shoonyatst(SHLIST), '-trade login-')

        if event == '-trade login-':
            if values[event] != -99:
                with open('data\\shoonya_params.txt','w') as file:
                    fp.write(','.join(SHLIST))
            else:
                cp('ERRORE,errore-->Error logging in', c='red on black')
            #sg.popup_scrolled(values[event], size=(80, None))


        def shoonyatst(configz):
            try:
                credentials = Shoonya.login_and_get_authorizations(username=configz[0], password=configz[1], panno=configz[2])
                    
                access_token = credentials["access_token"]
                key = credentials['key']
                token = credentials["token_id"]
                username = credentials['user_id']
                usercode = credentials["usercode"]
                usertype = credentials["usertype"]
                panno = credentials['panno']
                #shoonya = ShoonyaApi(configz[0], configz[1], configz[2],debug=True)
                
                shn = Shoonya(username, access_token, panno, key, token, usercode, usertype)
                bal = shn.get_limits()
                return bal
            except Exception as e:
                print(e)
                return -99


'''
