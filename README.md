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
A chain of basic stock screening scripts tied together with pysimplegui.

Scrapes daily NSE (volume/gainers/losers) stocks and generates picks based on (DEMA/Momentum/Price action) signals.

Generates weights for optimal portfolios via random weights and calculates/plots the efficient frontier for a mix of stocks.

Price alerts for watchlist stocks.

#Installation

Create a new python virtual environment and install requirements.txt :

`
python -m venv <name>
cd <name>
pip install -r requirements.txt
`
Note - python-Levenshtein might be a hassle to install for windows users; a simpler way would be to download the wheel from http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-levenshtein

... and install via 

`
pip install <path_to_whl>
`
Alternately, download the win 64 pyinstaller dist from the releases section.
