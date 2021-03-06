# yass
```
 _  _  __ _  ___ ___
| || |/ _` |(_-<(_-<
 \_, |\__,_|/__//__/
 |__/               
▌│█║▌║▌║ ***
*** ║▌║▌║█│▌
yₑₜ aₙₒₜₕₑᵣ ₛₜₒcₖ ₛcᵣₑₑₙₑᵣ
```
A chain of basic stock screening scripts (loosely) tied together with pysimplegui.

Scrapes daily NSE (volume/gainers/losers) stocks and generates picks based on (DEMA/Momentum/Price action) signals.

Generates weights for optimal portfolios via random weights and calculates/plots the efficient frontier for a mix of stocks.

Price alerts for watchlist stocks.

# Installation

(pls) install PressStart2P and OCR A Std fonts (for the gui and matplotlib):

https://fonts.google.com/specimen/Press+Start+2P

https://fontsgeek.com/fonts/OCR-A-Std-Regular

Create a new python virtual environment and install requirements.txt :

```
python -m venv <name>

cd <name>

pip install -r requirements.txt
```

Note - python-Levenshtein might be a hassle to install for windows users; a simpler way would be to download the wheel from http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-levenshtein

... and install via 


`
pip install <path_to_whl>
`

Alternately, download the win 64 pyinstaller dist from the releases section.

# Functions

1. Picks - day (interval 1d, period ytd) / iday (interval 1d, period ytd) : Generates daily stock pick shortlists.
2. BHS - buy/hodl/sell signals based on dema/momentum/price action (day/iday as above). 
3. Folio - gets the optimal weights of a mix of tickers (via text list/csv file upload)
4. Alerts - alerts for stocks added to watchlist (5 min alerts during market hours, 15 during the rest)
5. Del - remove stocks from watchlist alerts

Exports results to .csv (of course).

# Screenshots
![gif_all](https://github.com/squidinkscape/yass/blob/main/screenshots/demo.gif)
![main](https://github.com/squidinkscape/yass/blob/main/screenshots/screen_main.png)
![bhs](https://github.com/squidinkscape/yass/blob/main/screenshots/screen_bhs_rez.png)
![alerts-in](https://github.com/squidinkscape/yass/blob/main/screenshots/screen_alerts.png)
![alerts-out](https://github.com/squidinkscape/yass/blob/main/screenshots/screen_alerts_out.png)

# References

Most of the code is based off Sentdex's tutorials 🤘

https://pythonprogramming.net/finance-tutorials/

Yves Hilipisch's book 'Python for algorithmic trading' 

https://www.oreilly.com/library/view/python-for-algorithmic/

... and the odd medium article or two:

https://towardsdatascience.com/building-a-higher-dimensional-efficient-frontier-in-python-ce66ca2eff7

