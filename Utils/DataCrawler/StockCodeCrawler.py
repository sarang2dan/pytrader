import requests
from bs4 import BeautifulSoup
import sqlite3
import random
from logger import logger
import Utils.DataTypes as DataType
from Utils.DataTypes import *

class StockCodeCrawler(object):
    def downStockCode(self, marketType):

        url = 'http://datamall.koscom.co.kr/servlet/infoService/SearchIssue'
        html = requests.post(url,
                             data={'flag': 'SEARCH',
                                   'marketDisabled': 'null',
                                   'marketBit': marketType.value })
        return html.content

    def parseCodeFromHTML(self, html, stockInfo, marketType):
        soup = BeautifulSoup(html)
        options = soup.findAll('option')

        for opt in options:
            if len(opt) == 0:
                continue

            code = opt.text[1:7]

            companyName = opt.text[8:]
            isDelisted = 0
            if companyName.find('(폐지)') > -1:
                isDelisted = 1

            fullCode = opt.get('value')

            stockInfo.add(code, fullCode, companyName, marketType.value, isDelisted)
            pass

        return

class StockInfo(object):
    def __init__(self):
        self.items = {}
        return

    def count(self):
        return len(self.items)

    def clear(self):
        self.items.clear()
        self.__init__();
        return

    def add(self,
            code,
            fullCode,
            companyName,
            marketTypeNum,
            isDelisted,
            lastUpdate = None,
            startDate = None,
            endDate = None ):

        item = StockInfoItem.createItem(code,
                                        fullCode,
                                        companyName,
                                        marketTypeNum,
                                        isDelisted,
                                        lastUpdate,
                                        startDate,
                                        endDate)
        self.items[code] = item
        return

    def remove(self, stockCode):
        if self.items.get(stockCode) != None:
            del self.items[stockCode]
        return

    def find(self, stockCode):
        return self.items[stockCode]

    def dump(self):
        logger.info('-' * 50 )
        for i in self.items.values():
            logger.info( "Code: %s, ComName: %-15s (%6s), UpdateTime: %s, start: %s, end: %s" %
                   (StockInfoItem.getCode(i),
                    StockInfoItem.getCompanyName(i),
                    MARKET_TYPE.valueToName(StockInfoItem.getMarketType(i)),
                    StockInfoItem.getLastUpdate(i),
                    StockInfoItem.getStartDate(i),
                    StockInfoItem.getEndDate(i))
            )
        logger.info( "Total Company Count: %d\n" % len(self.items) )
        logger.info('-' * 50 )
        return

    def createTable(self, dbcon):
        sqlCreate = sql_stockInfo_schema
        try:
            dbcon.execute(sqlCreate)
        except:
            pass
        return

    def dropTable(self, dbcon):
        sqlDrop = "DROP TABLE stockInfo"
        try:
            dbcon.execute(sqlDrop)
        except:
            pass
        return

    def writeToDatabase(self, dbcon):
        sqlInsert = 'INSERT INTO stockInfo values( ?, ?, ?, ?, ?, ?, ?, ?)'

        for i in self.items.values():
            dbcon.execute(sqlInsert, i);

        dbcon.commit()
        return

    def loadFromDatabase(self, dbcon, marketType=MARKET_TYPE.NONE, except_delisted=True):
        sqlSelect = "SELECT * FROM stockInfo ORDER BY code ASC"
        cursor = dbcon.execute(sqlSelect)
        for i in cursor:
            if except_delisted == True:
                if StockInfoItem.getIsDelisted(i) == 1:
                    continue

            if marketType == MARKET_TYPE.NONE:
                self.items[StockInfoItem.getCode(i)] = i
            else:
                if StockInfoItem.getMarketType(i) == marketType.value:
                    self.items[StockInfoItem.getCode(i)] = i
        cursor.close()
        return

    def copyRandomStockCodes(self, item_count=100):
        if len(self.items) < item_count:
            logger.warn("[A count of stock items is %d, but needed %d items]" % (len(self.items, item_count)))
            count = len(self.items)

        codes = list(self.items.keys())
        rand = random.Random()
        rand.shuffle(codes)

        return codes[0:item_count]
    #
    # '''
    # def getColumn_LastUpdate(self, code):
    #     item = self.items[code]
    #     if item == None:
    #         return None
    #     return datetime.datetime.strptime(StockInfoItem.getLastUpdate(item), '%Y-%m-%d %H:%M:%S')
    #
    # def getColumn_StartDate(self, code):
    #     item = self.items.get(code)
    #     if item == None:
    #         return None
    #     return datetime.datetime.strptime(StockInfoItem.getStartDate(item), '%Y-%m-%d %H:%M:%S')
    #
    # def getColumn_EndDate(self, code):
    #     item = self.items.get(code)
    #     if item == None:
    #         return None
    #     return datetime.datetime.strptime(StockInfoItem.getEndDate(item), '%Y-%m-%d %H:%M:%S')
    # '''
    def updateStockDate(self, dbcon, code, lastUpdate, startDate, endDate):
        temp = list(self.items[code])
        temp[STOCK_INFO.lastUpdate.value] = str(lastUpdate)
        temp[STOCK_INFO.startDate.value]  = str(startDate)
        temp[STOCK_INFO.endDate.value]    = str(endDate)
        self.items[code] = tuple(temp)

        sqlDelete = "DELETE FROM stockInfo WHERE code ='%s'" % code
        sqlInsert = 'INSERT INTO stockInfo values( ?, ?, ?, ?, ?, ?, ?, ?)'
        cur = dbcon.cursor()
        cur.execute(sqlDelete)
        cur.execute(sqlInsert, temp);
        dbcon.commit()
        cur.close()
        return

'''
class StockCodeItem(object):
    marketType  = -1
    code        = ''
    fullCode    = ''
    companyName = ''

    def __init__(self, marketType, code, fullCode, companyName ):
        self.marketType = marketType
        self.code = code
        self.fullCode = fullCode
        self.companyName = companyName
        return
'''

if __name__ == '__main__':
    from Utils.DBManager import *
    import datetime

    DBManager.init()
    stockInfo = StockInfo()

    # scCrawler = StockCodeCrawler()
    # stockListHtml = scCrawler.downStockCode(MARKET_TYPE.KOSPI);
    # scCrawler.parseCodeFromHTML(stockListHtml, stockInfo, MARKET_TYPE.KOSPI)
    #
    # stockListHtml = scCrawler.downStockCode(MARKET_TYPE.KOSDAQ);
    # scCrawler.parseCodeFromHTML(stockListHtml, stockInfo, MARKET_TYPE.KOSDAQ)

    con = DBManager.getStockInfoDBConnection()

    # stockInfo.dropTable(con)
    # stockInfo.createTable(con)
    # stockInfo.writeToDatabase(con)


    stockInfo_test = StockInfo()
    stockInfo_test.loadFromDatabase(con, MARKET_TYPE.KOSDAQ)

    code = '074000'
    stockInfo_test.updateStockDate(con,
                                   code,
                                   datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                   datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    stockInfo_test.dump()

    print(str(StockInfoItem.getLastUpdate(stockInfo.items[code])))
    print(str(StockInfoItem.getStartDate(stockInfo.items[code])))
    print(str(StockInfoItem.getEndDate(stockInfo.items[code])))

    DBManager.finalize()

    del con
    del stockInfo_test
    # del stockInfo
    # del stockListHtml
    # del scCrawler