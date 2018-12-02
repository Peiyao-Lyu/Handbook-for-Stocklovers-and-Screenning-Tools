import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import pandas_datareader.data as web
import numpy as np
import pandas as pd
import matplotlib
import math
import matplotlib.pyplot as plt

def overview(ticker):
    link = 'https://finance.yahoo.com/quote/' + ticker + '/profile?p=' + ticker  
    response = requests.get(link)
    result_page = BeautifulSoup(response.content,'lxml')
    for tag in result_page.find_all('p',class_= "Mt(15px) Lh(1.6)"):
        print(tag.get_text())

def governance(ticker):
    link = 'https://finance.yahoo.com/quote/' + ticker + '/profile?p=' + ticker  
    response = requests.get(link)
    result_page = BeautifulSoup(response.content,'lxml')
    for tag in result_page.find_all('p',class_= "Fz(s)"):
        print(tag.get_text())

def summary(ticker):
	link = 'https://finance.yahoo.com/quote/' + ticker + '/?p' + ticker  
	response = requests.get(link)
	result_page = BeautifulSoup(response.content,'lxml')
	Summary_list = []
	for tag in result_page.find_all('td'):
		Summary_list.append(tag.get_text())

	values = list(Summary_list[1::2][i] for i in range(len(Summary_list[1::2])))
	indices = list(Summary_list[::2][i] for i in range(len(Summary_list[::2])))
	Summary_list = [indices,values]
	df = pd.DataFrame(Summary_list).transpose()
	df.columns = ["Index", "Value"]
	df.set_index("Index", inplace = True)
	return df

def ratios(ticker):
	link = 'https://finance.yahoo.com/quote/' + ticker + '/key-statistics?p=' + ticker      
	response = requests.get(link)
	result_page = BeautifulSoup(response.content,'lxml')
	VM_list = []
	for tag in result_page.find_all('td'):
		VM_list.append(tag.get_text())

	values = list(VM_list[1::2][i] for i in range(len(VM_list[1::2])))
	indices = list(VM_list[0::2][i] for i in range(len(VM_list[0::2])))

	sel_values = list(values[11:30][i] for i in range(len(values[11:30])))
	sel_indices = list(indices[11:30][i] for i in range(len(indices[11:30])))
	Finicial_ratio_list = [sel_indices,sel_values]

	df = pd.DataFrame(Finicial_ratio_list).transpose()
	df.columns = ["Finicial Ratio", "Value"]
	df.set_index("Finicial Ratio", inplace = True)
	return df

def executives(ticker):
	link = 'https://finance.yahoo.com/quote/' + ticker + '/profile?p=' + ticker  
	response = requests.get(link)
	result_page = BeautifulSoup(response.content,'lxml')
	key_executives_list = []

	for tag in result_page.find_all('td'):
		key_executives_list.append(tag.get_text())

	names = list(key_executives_list[::5][i] for i in range(len(key_executives_list[::5])))
	titles = list(key_executives_list[1::5][i] for i in range(len(key_executives_list[1::5])))
	pays = list(key_executives_list[2::5][i] for i in range(len(key_executives_list[2::5])))
	exerciseds = list(key_executives_list[3::5][i] for i in range(len(key_executives_list[3::5])))
	year_borns = list(key_executives_list[4::5][i] for i in range(len(key_executives_list[4::5])))
	key_executives_list = [names,titles,pays,exerciseds,year_borns]

	df = pd.DataFrame(key_executives_list).transpose()
	df.columns = ["Name", "Title", "Pay", "Exercised", "Year Born"]
	df.set_index ("Name", inplace = True)

	return df

def financial(ticker,IS=0,BS=0,CF=0):
    IS_keyword = ['Revenue','Operating Expenses','Income from Continuing Operations','Non-recurring Events','Net Income']
    BS_keyword = ['Assets','Liabilities','Equity']
    CF_keyword = ['Net Income','Operating Activities','Investing Activities','Financing Activities','Cash']
    link = 'https://finance.yahoo.com/quote/' + ticker + '/financials?p=' + ticker
    keyword = IS_keyword
    if BS:
        link = 'https://finance.yahoo.com/quote/' + ticker + '/balance-sheet?p=' + ticker
        keyword = BS_keyword
    if CF:
        link = 'https://finance.yahoo.com/quote/' + ticker + '/cash-flow?p=' + ticker
        keyword = CF_keyword
    response = requests.get(link)
    result_page = BeautifulSoup(response.content,'lxml')
    df_list = []
    fin_info = dict()
    col,data_col,data_frame = [None], [], []
    i = 1
    end = 0
    for tr_tag in result_page.find_all('tr'):
        for td_tag in tr_tag.find_all('td'):
            if td_tag.get('class') == ['C($gray)', 'Ta(end)']:
                col.append(td_tag.get_text())
            if not td_tag.get('class') in [['Fw(b)', 'Fz(15px)'],['C($gray)', 'Ta(end)'],['Fw(b)', 'Fz(15px)', 'Pb(8px)', 'Pt(36px)']]:
                if 1 <= i <= 5:
                    data_col.append(td_tag.get_text())
                    i += 1
                else:
                    data_frame.append(data_col)
                    data_col = []
                    i = 1
                    data_col.append(td_tag.get_text())
                    i += 1
            if tr_tag.get('class') == ['Bdbw(0px)!', 'H(36px)']:
                end += 1
            if end == 5:
                data_frame.append(data_col)
                df = pd.DataFrame(data_frame)
                df_list.append(df)
                end = 0
                i = 1
                data_col,data_frame = [], []
    for k in range(len(keyword)):
        col_name = col
        col_name[0] = keyword[k] + ' (All numbers in thousands)'
        df_list[k].columns = col_name
        df_list[k].set_index(col_name[0],inplace=True)
        fin_info[keyword[k]] = df_list[k]
    return (df_list, fin_info)

def IS(ticker):
    for table in financial(ticker)[0]:
        print(table)
    print(' ')
    print('For separate tables, please use other attributes, e.g. "revenue".')

def BS(ticker):
        for table in financial(ticker,BS=1)[0]:
            print(table)
        print(' ')
        print('For separate tables, please use other attributes, e.g. "assets".')

def CF(ticker):
    for table in financial(ticker,CF=1)[0]:
        print(table)
    print(' ')
    print('For separate tables, please use other attributes, e.g. "CFO".')

def revenue(ticker):
	return financial(ticker)[1]['Revenue']

def operating_expenses(ticker):
	return financial(ticker)[1]['Operating Expenses']

def operating_income(ticker):
    return financial(ticker)[1]['Income from Continuing Operations']

def non_recurring(ticker):
    return financial(ticker)[1]['Non-recurring Events']

def net_income(ticker):
    return financial(ticker)[1]['Net Income']

def assets(ticker):
    return financial(ticker,BS=1)[1]['Assets']

def liabilities(ticker):
    return financial(ticker,BS=1)[1]['liabilities']

def equity(ticker):
    return financial(ticker,BS=1)[1]['Equity']

def CFO(ticker):
    return financial(ticker,CF=1)[1]['Operating Activities']

def CFI(ticker):
    return financial(ticker,CF=1)[1]['Investing Activities']

def CFF(ticker):
    return financial(ticker,CF=1)[1]['Financing Activities']

def cash(ticker):
    return financial(ticker,CF=1)[1]['Cash']

def complete_analysis(ticker):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    link = 'https://finance.yahoo.com/quote/' + ticker + '/analysis?p' + ticker
    response = requests.get(link)
    if response.status_code == 200:
        pass
    else:
        print('An error occurred when accessing the page!')
    result_page = BeautifulSoup(response.content,'lxml')
    df_list = []
    analysis_info = dict()
    for table_tag in result_page.find_all('table'):
        col,data_col,data_frame = [], [], []
        for th_tag in table_tag.find_all('th'):
            col.append(th_tag.get_text())
        i = 1
        for td_tag in table_tag.find_all('td'):
            if 1 <= i <= 5:
                data_col.append(td_tag.get_text())
                i += 1
            else:
                data_frame.append(data_col)
                data_col = []
                i = 1
                data_col.append(td_tag.get_text())
                i += 1
        data_frame.append(data_col)
        df = pd.DataFrame(data_frame,columns = col)
        df.set_index(col[0],inplace=True)
        df_list.append(df)
        analysis_info[col[0]] = df
    return (df_list, analysis_info)

def analysis(ticker):
    for table in complete_analysis(ticker)[0]:
        print(table)
    print(' ')
    print('For separate tables, please use other attributes, e.g. "est_earnings".')

def est_earnings(ticker):
    return complete_analysis(ticker)[1]['Earnings Estimate']

def est_revenue(ticker):
    return complete_analysis(ticker)[1]['Revenue Estimate']
    
    
def history_earnings(ticker):
    return self.complete_analysis(ticker)[1]['Earnings History']
    
    
def EPS_trend(ticker):
    return complete_analysis(ticker)[1]['EPS Trend']
    
def EPS_revision(ticker):
    return complete_analysis(ticker)[1]['EPS Revisions']
    
def est_growth(ticker):
    return complete_analysis(ticker)[1]['Growth Estimates']

def holders_structure(ticker):
    link = 'https://finance.yahoo.com/quote/' + ticker + '/holders?p' + ticker
    response = requests.get(link)
    result_page = BeautifulSoup(response.content,'lxml')
    df_list = []
    for div_tag in result_page.find_all('div'):
        if div_tag.get('class') in [['Mt(25px)', 'Ovx(a)', 'W(100%)'],['W(100%)', 'Mb(20px)']]:
            if div_tag.find('span').get_text() == 'Breakdown':
                j = 2
                col = ['Major Holders Breakdown',' ']
            else:
                j = 5
                col = []
            i = 1
            data_col,data_frame = [],[]
            for th_tag in div_tag.find_all('th'):
                col.append(th_tag.get_text())
            for td_tag in div_tag.find_all('td'):
                if 1 <= i <= j:
                    data_col.append(td_tag.get_text())
                    i += 1
                else:
                    data_frame.append(data_col)
                    data_col = []
                    i = 1
                    data_col.append(td_tag.get_text())
                    i += 1
            data_frame.append(data_col)
            df = pd.DataFrame(data_frame,columns = col)
            df.set_index(col[0],inplace=True)
            df_list.append(df)
    df_list[1].index.names = ['Top Institutional Holders']
    df_list[2].index.names = ['Top Mutual Fund Holders']
    holders_info = {'major':df_list[0],'institutional':df_list[1],'mutual fund':df_list[2]}
    return (df_list,holders_info)

def holders(ticker):
    for table in holders_structure(ticker)[0]:
        print(table)
    print(' ')
    print('For separate tables, please use other attributes, e.g. "major_holders".')

def major_holders(ticker):
    return holders_structure(ticker)[1]['major']
    
    
def ins_holders(ticker):
    return holders_structure(ticker)[1]['institutional']

def mf_holders(ticker):
    return holders_structure(ticker)[1]['mutual fund']

def trading(ticker,sd,ed):
    start_date = sd
    start_list = start_date.split('-')
    start = datetime.datetime(int(start_list[0]),int(start_list[1]),int(start_list[2]))
    end_date = ed
    end_list = end_date.split('-')
    end = datetime.datetime(int(end_list[0]),int(end_list[1]),int(end_list[2]))
    df = web.DataReader(ticker,'yahoo',start_date,end_date)
    return df

def SMA(ticker, sd, ed, window):
    df = trading(ticker,sd,ed)
    df['Normal Return'] = df['Adj Close'].pct_change()
    df[str(window) + ' SMA'] = df['Adj Close'].rolling(window).mean()
    df['Signal'] = np.where(df[str(window) + ' SMA'] > df['Adj Close'],1,0)
    df['Log Return'] = df['Normal Return'].apply(lambda x:math.log(x+1))
    df['Strat Daily Return'] = np.where(df['Signal'] == 1, -df['Log Return'],0)
    df['Cumulative Sum'] = df['Strat Daily Return'].cumsum()
    plt.plot(df.reset_index()['Date'],df['Adj Close'], label = 'Close Price')
    plt.plot(df.reset_index()['Date'],df[str(window) + ' SMA'], label = str(window) + ' SMA')
    plt.legend()
    plt.title('Cumulative Return in %')
    plt.show()
    if df['Cumulative Sum'][-1] >= 0:
        print('Recomendation:')
        print('The simple moving average startegy of window ' + str(window) + ' yields a positive cumulative return. Hence, the strategy could be taken into consideration.')
    else:
        print('Recommendation:')
        print('The simple moving average startegy of window ' + str(window) + ' yields a negative cumulative return. Hence, the strategy should not be taken into consideration.')

    return df,df['Cumulative Sum'][-1]

def EMA(ticker, sd, ed, window):
    df = trading(ticker,sd,ed)
    df[str(window) + ' EMA'] = pd.Series.ewm(df['Adj Close'], span=window).mean()
    df['Signal'] = np.where(df[str(window) + ' EMA'] > df['Adj Close'],1,0)
    df['Normal Return'] = df['Adj Close'].pct_change()
    df['Log Return'] = df['Normal Return'].apply(lambda x:math.log(x+1))
    df['Strat Daily Return'] = np.where(df['Signal'] == 1, -df['Log Return'],0)
    df['Cumulative Sum'] = df['Strat Daily Return'].cumsum()
    plt.plot(df.reset_index()['Date'],df['Adj Close'], label = 'Close Price')
    plt.plot(df.reset_index()['Date'],df[str(window) + ' EMA'], label = str(window) + ' EMA')
    plt.legend()
    plt.title('Cumulative Return in %')
    plt.show()
    if df['Cumulative Sum'][-1] >= 0:
        print('Recomendation:')
        print('The exponential moving average startegy of window ' + str(window) + ' yields a positive cumulative return. Hence, the strategy could be taken into consideration.')
    else:
        print('Recommendation:')
        print('The exponential moving average startegy of window ' + str(window) + ' yields a negative cumulative return. Hence, the strategy should not be taken into consideration.')
    return df,df['Cumulative Sum'][-1]

def MR(ticker, sd, ed,s_window,l_window):
    df = trading(ticker,sd, ed)
    df['Log Price'] = df['Adj Close'].apply(lambda x:math.log(x))
    df['Normal Return'] = df['Adj Close'].pct_change()
    df['Log Return'] = df['Normal Return'].apply(lambda x:math.log(x+1))
    df[str(l_window) + ' SMA'] = df['Adj Close'].rolling(l_window).mean()
    df[str(s_window) + ' SMA'] = df['Adj Close'].rolling(s_window).mean()
    df['Long Signal'] = np.where(df[str(s_window) + ' SMA'] > df[str(l_window) + ' SMA'],1,0)
    df['Short Signal'] = np.where(df[str(s_window) + ' SMA'] < df[str(l_window) + ' SMA'],1,0)
    df['Strat Daily Return'] = np.where((df['Long Signal'] == 1)&(df['Short Signal'] == 0),df['Log Return'],
                                        np.where((df['Long Signal'] == 0)&(df['Short Signal'] == 1), -df['Log Return'],0))
    df['Cumulative Sum'] = df['Strat Daily Return'].cumsum()
    plt.plot(df.reset_index()['Date'],df['Adj Close'], label = 'Close Price')
    plt.plot(df.reset_index()['Date'],df[str(l_window) + ' SMA'], label = str(l_window) + ' SMA')
    plt.plot(df.reset_index()['Date'],df[str(s_window) + ' SMA'], label = str(s_window) + ' SMA')
    plt.legend()
    #plt.title('Cumulative Return in %')
    plt.show()
    return df,df['Cumulative Sum'][-1]

def RSI(ticker, sd, ed, window):
    df = trading(ticker,sd, ed)
    prices = df['Adj Close'].values
    deltas = np.diff(prices)
    seed = deltas[:window+1]
    up = seed[seed>=0].sum()/window
    down = -seed[seed<0].sum()/window
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100./(1.+rs)

    for i in range(window, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(window-1) + upval)/window
        down = (down*(window-1) + downval)/window

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    df['RSI'] = rsi
    df['Normal Return'] = df['Adj Close'].pct_change()
    #df[str(n) + ' SMA'] = df['Adj Close'].rolling(n).mean()
    df['Signal'] = np.where(df['RSI'] > 70,1,np.where(df['RSI'] < 30, -1, 0))
    df['Log Return'] = df['Normal Return'].apply(lambda x:math.log(x+1))
    df['Strat Daily Return'] = np.where(df['Signal'] == 1, df['Log Return'], np.where(df['Signal'] == -1,-df['Log Return'], 0))
    df['Cumulative Sum'] = df['Strat Daily Return'].cumsum()
    plt.plot(df.reset_index()['Date'],df['RSI'], label = 'RSI')
    plt.plot(df.reset_index()['Date'],np.array([30 for i in range(len(df['RSI']))]), label = '30')
    plt.plot(df.reset_index()['Date'],np.array([70 for i in range(len(df['RSI']))]), label = '70')
    plt.title('RSI')
    plt.legend()
    plt.show()
    return df,df['Cumulative Sum'][-1]

def recommend(ticker,sd, ed):
    print('Enter the window for Simple Moving Average strategy: ')
    s_window = int(input())
    print('Enter the window for Exponential Moving Average strategy: ')
    e_window = int(input())
    print('Enter the short window for Mean Reversion strategy: ')
    ms_window = int(input())
    print('Enter the long window for Mean Reversion strategy: ')
    ml_window = int(input())
    print('Enter the window for Relative Strength Index strategy: ')
    r_window = int(input())
    
    s_df,s_return = SMA(ticker,sd,ed,s_window)
    e_df,e_return = EMA(ticker,sd,ed,e_window)
    m_df,m_return = MR(ticker,sd,ed,ms_window,ml_window)
    r_df,r_return = RSI(ticker,sd,ed,r_window)

    return_list = [s_return,e_return,m_return,r_return]
    output_df = pd.DataFrame(return_list)
    output_df.index = ['SMA','EMA','MR','RSI']
    output_df.columns = ['Cumulative Return']

    if max(return_list) < 0:
        print("All investment strategies yield negatiev return so we wouldn't recommend you to invest in this stock using any above strategy.")
    else:
        if s_return == max(return_list):
            print('Overall Recommendation: With given parameters input, we would recommend you Simple Moving Average strategy in this stock because it yields max return of ' + str(round(s_return,2)) + ' on historical data.')
        elif e_return == max(return_list):
            print('Overall Recommendation: With given parameters input, we would recommend you Exponential Moving Average strategy in this stock because it yields max return of ' + str(round(e_return,2)) + ' on historical data.')
        elif m_return == max(return_list):
            print('Overall Recommendation: With given parameters input, we would recommend you Mean Reversion strategy in this stock because it yields max return of ' + str(round(m_return,2)) + ' on historical data.')
        else:
            print('Overall Recommendation: With given parameters input, we would recommend you Relative Strength Index strategy in this stock because it yields max return of ' + str(round(r_return,2)) + ' on historical data.')

    return output_df

def index():
    import requests
    index_dict = dict()
    url = "https://www.briefing.com/investor/markets/stock-market-update/"
    response = requests.get(url)
    results_page = BeautifulSoup(response.content,'lxml')
    all_td_tags = results_page.find_all('td', valign="top")
    count = 1
    index_list = []
    for tag in all_td_tags:
        if count % 4 == 1:
            index_name = tag.get_text()
        elif count % 4 == 2:
            index_data = tag.get_text()
        elif count % 4 == 3:
            index_change = tag.get_text()
        elif count % 4 == 0:
            index_change_pct = tag.get_text().strip()
            index_change_pct = index_change_pct.replace('(','')
            index_change_pct = index_change_pct.replace(')','')
            index_change_pct = index_change_pct.replace('%','')
            index_tuple = (index_name,index_data,index_change,index_change_pct)
            index_list.append(index_tuple)
        if count == 12:
            break
        count+=1

    index_name_list = [index_list[i][0] for i in range(3)]
    index_data_list = [index_list[i][1] for i in range(3)]
    index_change_list = [index_list[i][2] for i in range(3)]
    index_change_pct_list = [index_list[i][3] for i in range(3)]
    index_ = ['Current Data', 'Change', '%Change']
    columns_ = index_name_list
    df = pd.DataFrame([index_data_list,index_change_list,index_change_pct_list], index = index_, columns = columns_)
    return df

def industry():
    index_dict = dict()
    url = "https://www.briefing.com/investor/markets/stock-market-update/"
    response = requests.get(url)
    results_page = BeautifulSoup(response.content,'lxml')
    all_td_tags = results_page.find_all('td', valign="top")
    count = 1
    strong_list = []
    weak_list = []
    for tag in all_td_tags:
        if count == 17:
            print(tag.get_text())
        if count == 18:
            print(tag.get_text())
        count += 1
        
def market_news():
    import requests
    import pandas as pd
    from bs4 import BeautifulSoup
    index_dict = dict()
    url = "https://www.briefing.com/investor/markets/stock-market-update/"
    response = requests.get(url)
    results_page = BeautifulSoup(response.content,'lxml')
    all_td_tags = results_page.find_all('td')
    count = 1
    for tag in all_td_tags:
        if count == 32:
            print("Moving the Market:")
            print(tag.get_text())
        count+=1

def SNP(start,end):
    df_SNP = web.DataReader('^GSPC', 'yahoo', start=start, end=end)['Adj Close']
    df_SNP.plot(title = "S&P 500 Historical Data")
  
def DJI(start,end):
    df_DJI = web.DataReader('^DJI', 'yahoo', start=start, end=end)['Adj Close']
    df_DJI.plot(title = "DJI Historical Data")

def IXIC(start,end):
    df_IXIC = web.DataReader('^IXIC', 'yahoo', start=start, end=end)['Adj Close']
    df_IXIC.plot(title = 'IXIC Historical Data')

def best_performers(x=1):
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt

    output_list = []
    stock_list = []
    pct_list = []
    freq_list=[]
    new_list=[]
    cloud_list=[]

    if x==1:
        url = "https://csimarket.com/markets/Stocks.php?days=yday&=#tablecomp"
    elif x==2:
        url = "https://csimarket.com/markets/Stocks.php?days=week&=#tablecomp"
    elif x==3:
        url = "https://csimarket.com/markets/Stocks.php?days=month&=#tablecomp"
    elif x==4:
        url = "https://csimarket.com/markets/Stocks.php?days=quarter&=#tablecomp"
    elif x==5:
        url = "https://csimarket.com/markets/Stocks.php?days=ytd&=#tablecomp"
    else:
        return 'Wrong input!'

    results_page = requests.get(url)
    soup = BeautifulSoup(results_page.content,'lxml')
    all_tags = soup.find_all('td', class_='lk')
    for tag in all_tags:
        output_list.append(tag.get_text().replace(',','').replace('\xa0','').replace(' ',''))
    for i in range(0,40):
        if i % 2 == 0:
            stock_list.append(output_list[i])
        if i % 2 == 1:
            pct_list.append((output_list[i]))

    df = pd.DataFrame(pct_list, stock_list)    
    if x==1:
        df.index.name = 'Daily Best Performing Stocks'
    elif x==2:
        df.index.name = 'Weekly Best Performing Stocks'
    elif x==3:
        df.index.name = 'Monthly Best Performing Stocks'
    elif x==4:
        df.index.name = 'Quarterly Best Performing Stocks'
    elif x==5:
        df.index.name = 'Year-to-Date Best Performing Stocks'
    else:
        return None
    df.columns = ['pct_change']

    for i in pct_list:
        x=round(float(i.replace('%','')))
        freq_list.append(x)
    new_list=stock_list+freq_list
    for i in range(0,20):
        for x in range(0,new_list[i+20]):
            cloud_list.append(new_list[i])

    stock_string=(" ").join(cloud_list)
    wordcloud = WordCloud(width = 1600, height = 800, relative_scaling=0.5, max_words = 1000, max_font_size=60, min_font_size=10, 
                        background_color="white", repeat=True, margin=50, random_state=3).generate(stock_string)
    plt.figure(figsize=(20,10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    #plt.savefig("wordcloud"+".png", bbox_inches='tight')
    plt.show()
    plt.close()

    return df

def major_index():
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd

    index_list = []
    last_list = []

    url = "https://markets.wsj.com/"
    results_page = requests.get(url)
    soup = BeautifulSoup(results_page.content,'lxml')
    index_tags = soup.find_all('td', class_='firstCol')
    last_tags = soup.find_all('td', class_='dataCol dataColCenter')

    for tag in index_tags:
        index_list.append(tag.get_text().replace('\n',''))
    for tag in last_tags:
        last_list.append(tag.get_text())
    index_list = index_list[23:42]

    df = pd.DataFrame(last_list, index_list)
    df.index.name = 'Major U.S. Stock Indexes'
    df.columns = ['Last']

    return df


def forex():
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd

    cu_list = []
    url = "https://markets.on.nytimes.com/research/markets/currencies/currencies.asp"
    results_page = requests.get(url)
    soup = BeautifulSoup(results_page.content,'lxml')
    cu_tags = soup.find_all('td')

    for tag in cu_tags:
        cu_list.append(tag.get_text())
    cu_list=cu_list[60:90]
    df = pd.DataFrame(cu_list[1::6], columns=['Dollar'])
    df.rename(index={0:cu_list[::6][0],1:cu_list[::6][1],2:cu_list[::6][2],3:cu_list[::6][3],4:cu_list[::6][4]}, inplace=True)
    df['Euro'] = cu_list[2::6]
    df['Pound'] = cu_list[3::6]
    df['Yen'] = cu_list[4::6]
    df['Yuan'] = cu_list[5::6]

    return df

def futures():
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd
    import re

    name=[]
    name1=[]
    price=[]
    price1=[]
    price2=[]
    change=[]
    pctchange=[]

    pattern = r'\d+.\d+'
    url = "https://www.marketwatch.com/tools/futures"
    results_page = requests.get(url)
    soup = BeautifulSoup(results_page.content,'lxml')
    l1_tags = soup.find_all('td',class_='name')
    l2_tags = soup.find_all('td',class_='price')
    l3_tags = soup.find_all('td',class_='bgChange')
    l4_tags = soup.find_all('td',class_='bgPercentChange')

    for tag in l1_tags:
        name.append(tag.get_text())   
    for string in name:
        l = string.split('/')
        if 'quotes' in l:
            for i in range(0, l.index('quotes')):
                name1.append(l[i])
        else:
            name1.append(string)

    for tag in l2_tags:
        price.append(tag.get_text().replace('\n',''))

    for tag in l3_tags:
        change.append(tag.get_text())

    for tag in l4_tags:
        pctchange.append(tag.get_text())

    df = pd.DataFrame(price, columns=['Last'])
    for i in range(0,len(name1)):
        df.rename(index={i: name1[i]}, inplace=True)
    df['Change'] = change
    df['Change %'] = pctchange
    df.index.name = 'Futures Contract Names'
    return df