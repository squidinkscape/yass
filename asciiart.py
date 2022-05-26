from art import *
import sys
import logging
import random

logger = logging.getLogger(__name__)


LOGO = text2art('yass',font='fancy23',decoration='barcode1')

MAIN_MENU = text2art('''
yass
''', font='small')+text2art('''***
***''', font='fancy2', decoration= 'barcode1') + ('\n')+text2art(f'yet another stock screener', font='subscript1')+('\n')

DAILYLOGO = text2art('picks',font='fancy23',decoration='barcode1')

SAVELOGO = text2art('save?',font='fancy23',decoration='barcode1')

LISTLOGO = text2art('list',font='fancy23',decoration='barcode1')

CSVLOGO = text2art('list',font='fancy23',decoration='barcode1')

logger.info(f'--->{__name__} loaded<----')
