
# todo: 파일이름을 commons.py로 변경하자

from enum import Enum
import datetime

class STOCK_INFO(Enum):
    code = 0
    fullcode = 1
    companyName = 2
    marketType = 3
    isDelisted = 4
    lastUpdate = 5
    startDate = 6
    endDate = 7
    pass

sql_stockInfo_schema = \
    """CREATE TABLE stockInfo ( {} char(8) primary key,
                                {} char(12),
                                {} char(64),
                                {} integer,
                                {} integer,
                                {} date,
                                {} date,
                                {} date )""".format(*STOCK_INFO.__dict__['_member_names_'])

class STOCK_PRICE(Enum):
    Date = 0
    Open = 1
    High = 2
    Low = 3
    Close = 4
    Volume = 5
    def valueToName(num):
        return MARKET_TYPE.__dict__['_value2member_map_'][num].name
    pass

sql_stockPrice_schema = \
    """CREATE TABLE %s (
            {} date primary key,
            {} integer,
            {} integer,
            {} integer,
            {} integer,
            {} integer
        )""".format(*STOCK_PRICE.__dict__['_member_names_'])

class MARKET_TYPE(Enum):
    """abc"""
    NONE = 0
    KOSPI = 1
    KOSDAQ = 2

    def valueToName(num):
        return MARKET_TYPE.__dict__['_value2member_map_'][num].name
    pass

def convertStringToDateTime(strDate):
    """strDate should have %Y-%m-%d %H:%M:%S format"""
    return datetime.datetime.strptime( strDate, '%Y-%m-%d %H:%M:%S')

def getCurrentTime():
    t = datetime.datetime.now()
    return datetime.datetime(t.year,
                             t.month,
                             t.day,
                             t.hour,
                             t.minute,
                             t.second)

class StockInfoItem:
    def createItem(code,
                   fullCode,
                   companyName,
                   marketTypeNum,
                   isDelisted,
                   lastUpdate,
                   startDate,
                   endDate):
        return (code,
                fullCode,
                companyName,
                marketTypeNum,
                isDelisted,
                lastUpdate,
                startDate,
                endDate)
    def getCode(stock_info_item):
        return stock_info_item[STOCK_INFO.code.value]

    def getCompanyName(stock_info_item):
        return stock_info_item[STOCK_INFO.companyName.value]

    def getFullCode(stock_info_item):
        return stock_info_item[STOCK_INFO.fullcode.value]

    def getIsDelisted(stock_info_item):
        return stock_info_item[STOCK_INFO.isDelisted.value]

    def getMarketType(stock_info_item):
        return stock_info_item[STOCK_INFO.marketType.value]

    def getLastUpdate(stock_info_item):
        return stock_info_item[STOCK_INFO.lastUpdate.value]

    def getStartDate(stock_info_item):
        return stock_info_item[STOCK_INFO.startDate.value]

    def getEndDate(stock_info_item):
        return stock_info_item[STOCK_INFO.endDate.value]
    pass

def makePriceTableName(code, row_unit):
    return "price_%s_%s" % (code, row_unit)
'''
import sys
def getMethodName():
    return sys._getframe(1).f_code.co_name + "()"
'''

import inspect
def getMethodName():
    # caller의 이름을 리턴
    return inspect.stack()[1].function
