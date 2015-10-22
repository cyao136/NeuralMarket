'''
Created on Oct 22, 2015

@author: Chengyu Yao
@summary: Configuration for the clients
'''

from Pruner import yahoo_finance_client_pruner

class ClientConfig(object):
    url = None
    pruner = None

class YahooFinanceClientConfig(ClientConfig):
    url = None
    pruner = yahoo_finance_client_pruner