## Introduction:
**sft**, which stands for Simplified Finance Terminal, is a comprehensive python finance library. It aims to provide up-to-date information for finance enthusiasts. Users can directly access information of the market and public companies based on their needs. Users can apply the functions in sft to extract financial data, apply investment strategies, obtain financial statements, visualize market performance, and so on. 

## Project Group:
- Group name: SFT Group
- Group members: Ruofei Hu (rh2910), Fengfan Yang (fy2244), Wenyu Zhou (wz2445), Peiyao Lyu (pl2697)
- Section 2

## Strategies description:
The terminal includes 4 basic investment strategies derived from the following index: Simplified Moving Average (SMA), Exponential Moving Average (EMA), Mean Reversion (MR), Relative Strength Index (RSI).
- SMA strategy: Given window, calculate SMA and compare it with close price daily to choose buy, hold or sell.
- EMA strategy: Given window, calculate EMA and compare it with close price daily to choose buy, hold or sell.
- MR strategy: Given short and long windows, find short and long signals and use crossovers to choose buy, hold, or sell.
- RSI strategy: Given window, calculate RSI and use RSI to choose buy, hold, or sell.

## Tools we used:
- Object Oriented Programming (OOP)
- BeautifulSoup 
- Pandas
- Numpy
- etc.

## Run Instruction:
1. Import sft

![](https://github.com/Peiyao-Lyu/Handbook-for-Stocklovers-and-Screenning-Tools/blob/master/import.png)


2. Use methods according to user’s needs
![](https://github.com/Peiyao-Lyu/Handbook-for-Stocklovers-and-Screenning-Tools/blob/master/ex1.png)

![](https://github.com/Peiyao-Lyu/Handbook-for-Stocklovers-and-Screenning-Tools/blob/master/ex2.png)

## Documentations:
**overview**(ticker -> str):<br /> 
Return the overview of the company in string format. The argument must be a correct ticker of a listed company in string format.

**governance**(ticker -> str): Return the overview of the company’s corporate governance in string format, including different assessment scores. The argument must be a correct ticker of a listed company in string format.

**summary**(ticker -> str): 
Return a set of indices about the company’s stock price and fundamental financial ratios in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**ratios**(ticker -> str): 
Return a set of financial ratios of the company, including but not limited to profit margin, ROA, ROE, etc. in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**executives**(ticker -> str): 
Return a pandas DataFrame including the name, title, salary, exercised option and year of birth of the major executives of the company. The argument must be a correct ticker of a listed company in string format.

**IS**(ticker -> str): 
Print the income statement of the company in the last four fiscal periods (if available) in a list of pandas DataFrames. The argument must be a correct ticker of a listed company in string format.

**BS**(ticker -> str): 
Print the balance sheet of the company in the last four fiscal periods (if available) in a list of pandas DataFrames. The argument must be a correct ticker of a listed company in string format.

**CF**(ticker -> str): 
Print the statement of cash flows of the company in the last four fiscal periods (if available) in a list of pandas DataFrames. The argument must be a correct ticker of a listed company in string format.

**revenue**(ticker -> str): 
Return the revenues financials of the company in the most recent four accounting periods (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**operating_expenses**(ticker -> str): 
Return the operating expenses financials of the company in the most recent four accounting periods (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**non_recurring**(ticker -> str): 
Return the financials related to the non-recurring events of the company in the most recent four accounting periods (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**net_income**(ticker -> str): 
Return the net income financials of the company in the most recent four accounting periods (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.
assets(ticker -> str)
Return the assets financials of the company in the most recent four accounting periods (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**liabilities**(ticker -> str): 
Return the liabilities financials of the company in the most recent four accounting periods (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**equity**(ticker -> str): 
Return the equity financials of the company in the most recent four accounting periods (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**CFO**(ticker -> str): 
Return the information of the the company’s cash flow from operating activities in the most recent four accounting periods (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**CFI**(ticker -> str): 
Return the information of the the company’s cash flow from investing activities in the most recent four accounting periods (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**CFF**(ticker -> str): 
Return the information of the the company’s cash flow from financing activities in the most recent four accounting periods (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**cash**(ticker -> str):
Return the information of the the company’s change in cash and cash equivalents in the most recent four accounting periods (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**analysis**(ticker -> str): 
Print an overall analysis of the the company’s earning, revenues, EPS, etc. (if available) in a list of pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**est_earnings**(ticker -> str): 
Return the information of the the company’s earnings estimate (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**est_revenue**(ticker -> str): 
Return the information of the the company’s revenue estimate (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**history_earnings**(ticker -> str): 
Return the information of the the company’s history earnings (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**EPS_trend**(ticker -> str): 
Return the information of the the company’s EPS trend (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**EPS_revision**(ticker -> str): 
Return the information of the the company’s EPS revisions (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**est_growth**(ticker -> str): 
Return the information of the the company’s earnings estimate (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**holders**(ticker -> str): 
  Print the overall information of the the company’s holders (if available) in a list of pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**major_holders**(ticker -> str): 
  Return the information of the the company’s major holders  (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**ins_holders**(ticker -> str): 
  Return the information of the the company’s institutional holders (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**mf_holders**(ticker -> str): 
  Return the information of the the company’s mutual fund holders (if available) in a pandas DataFrame. The argument must be a correct ticker of a listed company in string format.

**trading**(ticker,sd,ed): 
  Return a dataframe containing price information of ticker from sd to ed
  Input format: start date = ‘YYYY-MM-DD’ , end date = ‘YYYY-MM-DD’
  
**SMA**(ticker, sd, ed, window):
	Show the plot of SMA and close price from sd to ed
  Return a dataframe with simple moving average and cumulative sum, and cumulative return in a tuple form
  Input format: sd = ‘YYYY-MM-DD’ , ed = ‘YYYY-MM-DD'

**EMA**(ticker, sd, ed, window):
	Show the plot of EMA and close price from sd to ed
  Return a dataframe with exponential moving average and cumulative sum, and cumulative return in a tuple form
  Input format: sd = ‘YYYY-MM-DD’ , ed = ‘YYYY-MM-DD’

**MR**(ticker, sd, ed,s_window,l_window):
	Show the plot of short-term and long-term SMAs and close price from sd to ed
  Return a dataframe with long signal, short signal and cumulative sum, and cumulative return in a tuple form
	Input format: sd = ‘YYYY-MM-DD’ , ed = ‘YYYY-MM-DD’

**RSI**(ticker, sd, ed, window):
	Show the plot of RSI from sd to ed
  Return a dataframe with RSI and cumulative sum, and cumulative return in a tuple form
  Input format: sd = ‘YYYY-MM-DD’ , ed = ‘YYYY-MM-DD’

**Recommend**(ticker,sd, ed):
	Show the plot of SMA and close price from sd to ed
	Show the plot of EMA and close price from sd to ed
	Show the plot of short-term and long-term SMAs and close price from sd to ed
  Show the plot of RSI from sd to ed
  Print out the best strategy for given stock
	Return a dataframe containing the cumulative return for each investment strategy
  Input format: sd = ‘YYYY-MM-DD’ , ed = ‘YYYY-MM-DD’

**index**():
Return a dataframe containing the current data, change, and percentage change for Nasdaq, S&P, DJIA

**industry**():
	Print out the current strong and weak industries

**market_news**():
	Print out the current market news

**SNP**(start,end):
	Show the plot of S&P from start to end

**DJI**(start,end):
	Show the plot of DJIA from start to end

**IXIC**(start,end):
	Show the plot of Nasdaq from start to end

**best_performers**(x=1): 
Returns a word cloud of the names of best performing stocks and a dataframe of these stocks and their percentage changes in a given period. For daily best performing stocks, input 1; weekly: 2; monthly: 3; quarterly: 4; yearly: 5.

**major_index**():
	Returns the names and the most current values of 19 major U.S. Stock indexes.

**forex**():
	Returns a most current cross currency table including U.S. Dollars, Euros, British Pounds, Japanese Yen, and Chinese Yuan.

**futures**():
	Returns a dataframe of 117 major futures contracts with up-to-date market prices, price changes, and price percentage changes.
