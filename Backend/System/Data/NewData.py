'''
Created on Oct 22, 2015

@author: Chengyu Yao
@summary: Script to get new data
'''
from CacheLibrary import cache_data
from YahooFinanceClient import YahooFinanceClient

if __name__ == "__main__":
    cache_data(YahooFinanceClient().get_data())