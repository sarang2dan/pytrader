# FinanceDataCrawler
# todo: Define Table schema ( logical and physical.. whatever )
# todo: 분단위, 일단위, 월단위, 일단 분단위는 어떻게 해야하나...
# todo: 최종적으로 분단위 트레이딩이다.

import datetime
import pandas_datareader.data as web
from pandas import DataFrame, Series

import aiohttp
import asyncio
import os.path
import re
import sys
import sqlite3
from bs4 import BeautifulSoup
from collections import namedtuple

from logger import *
from os import getpid
import warnings
import urllib
from urllib.request import urlopen

from Utils.DataCrawler.StockCodeCrawler import StockInfo
from Utils.DBManager import *
from Utils.DataTypes import *

class FinanceDataCrwaler(object):
    def createTable(self, dbcon, tableName):

        sql = sql_stockPrice_schema % tableName

        try:
            dbcon.execute(sql)
            dbcon.commit()
        except Exception as e:
            logger.error(str(e))
        return
    def dropTable(self, dbcon, tableName):
        sql = "DROP TABLE %s" % tableName
        try:
            dbcon.execute(sql)
            dbcon.commit()
        except Exception as e:
            logger.error("[%s:%s]"%(sql, str(e)))
        return

    # todo: 성능을 위해서 수정할 필요가 있다.
    def maybeQueryItem(self, dbStartDate, dbEndDate, qryStartDate, qryEndDate):
        shouldQuery = True
        startPos = 0 # left:1, in: 2 ,right: 3
        endPos= 0 # 1, 2, 3

        if dbStartDate > qryStartDate:
            startPos = 1
        elif dbEndDate <= qryStartDate:
            startPos = 3
        else:
            startPos = 2

        if dbEndDate < qryEndDate:
            endPos = 1
        elif dbStartDate >= qryEndDate:
            endPos = 3
        else:
            endPos = 2

        if startPos == 1:
            if endPos != 3:
                qryEndDate = dbEndDate
        elif startPos == 2:
            if endPos == 2:
                shouldQuery = False
            elif endPos == 3:
                qryStartDate = dbStartDate
            else:
                raise ValueError(
                    "Check the time: \n   dbStartDate: %s, dbEndDate: %s, qryStartDate: %s, qryEndDate: %s" %
                    (dbStartDate, dbEndDate, qryStartDate, qryEndDate))
        elif startPos == 3:
            if endPos == 3:
                qryStartDate == dbStartDate
            else:
                raise ValueError(
                    "Check the time: \n   dbStartDate: %s, dbEndDate: %s, qryStartDate: %s, qryEndDate: %s" %
                    (dbStartDate, dbEndDate, qryStartDate, qryEndDate))

        return shouldQuery, qryStartDate, qryEndDate
    pass

class DaumFinanceDataCrawler(FinanceDataCrwaler):
    pass


def getLastPage(code):
    url = 'http://finance.naver.com/item/sise_day.nhn?code=' + code
    html = urlopen(url)
    bs = BeautifulSoup(html.read(), "html.parser")
    lastpage = 1

    try:
        td_lastpage = bs.find_all("td", class_="pgRR")
        if len(td_lastpage) != 0:
            link_lastpage = td_lastpage[0].a.get('href')
            lastpage = int(re.search(r'page=(\d+)', link_lastpage).group(1))
        else:
            lastpage = 1
    except Exception as e:
        logger.warn(code)
        logger.error('%s %s [%d]' % (str(e), str(td_lastpage), lastpage))
    return lastpage

async def fetch(code, date_from, date_to, page, last_page):
    global semaphore
    with await semaphore:
        text = await request(code, date_from, date_to, page, last_page)
        if text is not None:
            prices = await parse(code, date_from, date_to, text, page, last_page)
            logger.info("fetching! [code:%s page:%d/%d]" % (code, page, last_page))
            await save(code, prices)
            del prices

        await asyncio.sleep(0.5)

    if len(asyncio.Task.all_tasks()) == 1:
        asyncio.get_event_loop().stop()
    pass

async def request(code, date_from, date_to, page, last_page):
    url = 'http://finance.naver.com/item/sise_day.nhn?code={}&page={}'.format(code, page)
    TIMEOUT = 4
    try:
        with aiohttp.Timeout(TIMEOUT):
            #async with aiohttp.ClientSession.request('GET', url) as resp:
            async with aiohttp.request('GET', url) as resp:
                assert resp.status == 200
                return await resp.text(encoding='euc-kr')
    except (asyncio.TimeoutError, Exception) as e:
         logger.warn("Timeout {} seconds from requesting(code: {})".format(TIMEOUT, code))
         logger.warn(e)

         asyncio.get_event_loop().create_task(fetch(code, date_from, date_to, page, last_page))
         return None

async def parse(code, date_from, date_to, text, page, last_page):
    logger.debug(
        "Parsing {:,d} bytes for ({}) on page {}".format(len(text), code, page))

    Price = namedtuple('Price', ['code', 'date', 'open', 'high', 'low', 'close', 'volume'])
    prices = []
    lastpage = 1

    date_str = None
    close = None
    gap = None
    open = None
    high = None
    low = None
    volume = None

    bs = BeautifulSoup(text, 'html.parser')
    trlists = bs.find_all("tr")

    for tr in trlists:
        try:
            if tr.span == None:
                continue
            '''
            logger.warn("1")

            #tds = trlists[i].find_all('td')
            tds = trlists[i].findAll('td')

            logger.warn("2")

            if tds == None or len(tds) != 7 or tds[0].text == '' or  tds[0].text == None:
                continue

            logger.warn("3")
            date = datetime.datetime.strptime(tds[0].text, '%Y.%m.%d')
            if date > date_to:
                continue
            logger.warn("4")

            if date < date_from:
                return prices

            logger.warn("5")

            date_str = str(date)
            logger.warn("7")

            close = int(tds[1].text.replace(',', ''))

            logger.warn("8")
            gap = int(tds[2].text.replace(',', ''))  # unused
            logger.warn("9")
            open = int(tds[3].text.replace(',', ''))
            logger.warn("10")
            high = int(tds[4].text.replace(',', ''))
            logger.warn("11")

            low = int(tds[5].text.replace(',', ''))
            logger.warn("12")

            volume = int(tds[6].text.replace(',', ''))
            logger.warn("6")
            '''

            # tds = tr.findAll('td', attrs={'class': 'num', 'align': 'center'})
            date_txt = tr.findAll("td",align="center")[0]
            tds = tr.findAll("td", class_="num")

            if len(date_txt.text) == 0:
                continue

            if len(tds) != 6:
                continue

            date = datetime.datetime.strptime(date_txt.text, '%Y.%m.%d')
            if date > date_to:
                continue
            if date < date_from:
                return prices

            date_str = str(date)
            close = int(tds[0].text.replace(',', ''))
            gap = int(tds[1].text.replace(',', ''))  # unused
            open = int(tds[2].text.replace(',', ''))
            high = int(tds[3].text.replace(',', ''))
            low = int(tds[4].text.replace(',', ''))
            volume = int(tds[5].text.replace(',', ''))

            prices.append(Price(code, date_str, open, high, low, close, volume))
        except (ValueError, IndexError) as e:
            logger.warn("Parsing error: code {} (page: {}) [{}]".format(code, page, str(e)))

        pass # for

    if page > 2:
        asyncio.get_event_loop().create_task(
            fetch(code, date_from, date_to, page-1, last_page))
    del bs

    return prices
'''
import urllib
import time

from urllib.request import urlopen
from bs4 import BeautifulSoup

stockItem = '005930'

url = 'http://finance.naver.com/item/sise_day.nhn?code='+ stockItem
html = urlopen(url)
source = BeautifulSoup(html.read(), "html.parser")

maxPage=source.find_all("table",align="center")
mp = maxPage[0].find_all("td",class_="pgRR")
mpNum = int(mp[0].a.get('href')[-3:])

for page in range(1, mpNum+1):
  print (str(page) )
  url = 'http://finance.naver.com/item/sise_day.nhn?code=' + stockItem +'&page='+ str(page)
  html = urlopen(url)
  source = BeautifulSoup(html.read(), "html.parser")
  srlists=source.find_all("tr")
  isCheckNone = None

  if((page % 1) == 0):
    time.sleep(1.50)

  for i in range(1,len(srlists)-1):
   if(srlists[i].span != isCheckNone):

    srlists[i].td.text
    print(srlists[i].find_all("td",align="center")[0].text, srlists[i].find_all("td",class_="num")[0].text )
'''

async def save(code, prices, row_unit='day'):
    global conn
    global inserted_rows

    cursor = conn.cursor()

    for p in prices:
        try:
            sqlInsert = "INSERT INTO %s VALUES (?, ?, ?, ?, ?, ?) " % makePriceTableName(code, row_unit)
            cursor.execute(sqlInsert, p[1:])  # code는 제외하고 insert
            inserted_rows += cursor.rowcount
        except (sqlite3.IntegrityError , ValueError ) as e:
            logger.warn(e)
            cursor.execute(
                """UPDATE {} SET date = '{p.date:%Y-%m-%d %H:%M:%S},
                                 open = {p.open},
                                 high = {p.high},
                                 low = {p.low},
                                 close = {p.close},
                                 volume = {p.volume}
                    WHERE code = '{p.code}'""".format(makePriceTableName(code, row_unit), p=p) )
        pass

    conn.commit()
    pass

class NaverFinanceDataCrawler(FinanceDataCrwaler):
    def updateAllStockPriceData(self,
                                stockDBCon: sqlite3.Connection,
                                priceDBCon: sqlite3.Connection,
                                stockInfo: StockInfo,
                                marketType,
                                start_date,
                                end_date,
                                row_unit='day'):
        # def main(self, stockDBCon, priceDBCon, stockInfo, start_date, end_date, symbols):
        global semaphore
        global conn
        global inserted_rows

        logger.info("Starting process (pid:{})".format(getpid()))
        inserted_rows = 0
        qryStartDate = None
        qryEndDate   = None

        msg = ''
        msg += '\n\n\n[%s] Start Downloading Stock Price Data\n' % getMethodName()
        msg += '                 start date: %s\n' % str(start_date.date())
        msg += '                 end date: %s\n' % str(end_date.date())
        logger.info(msg)

        totalCnt = len(stockInfo.items)
        curCnt = 0

        logger.debug('Preparing an event loop')

        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(10)
        conn = priceDBCon

        haveToQuery = True
        try:
            for i in stockInfo.items.values():
                shouldQuery = True
                curCnt += 1

                logger.debug("[%d/%d][code:%s]" % \
                             (curCnt, totalCnt, StockInfoItem.getCode(i)))

                if marketType.value != StockInfoItem.getMarketType(i):
                    continue

                if StockInfoItem.getIsDelisted(i) == 1:
                    continue

                qryStartDate = None
                qryEndDate   = None

                if StockInfoItem.getLastUpdate(i) == None:
                    shouldQuery = True
                    qryStartDate = start_date
                    qryEndDate = end_date
                    self.createTable(priceDBCon,
                                     makePriceTableName(StockInfoItem.getCode(i), row_unit=row_unit))
                else:
                    shouldQuery, qryStartDate, qryEndDate =\
                        self.maybeQueryItem(convertStringToDateTime(StockInfoItem.getStartDate(i)),
                                            convertStringToDateTime(StockInfoItem.getEndDate(i)),
                                            start_date,
                                            end_date)
                    pass # if getLastUpdate

                if shouldQuery == True:
                    last_page = getLastPage(StockInfoItem.getCode(i))
                    loop.create_task(fetch(StockInfoItem.getCode(i), qryStartDate, qryEndDate, last_page, last_page))
                    stockInfo.updateStockDate(stockDBCon,
                                              StockInfoItem.getCode(i),
                                              getCurrentTime(),
                                              qryStartDate,
                                              qryEndDate)
                    pass
                pass  # for i in stockInfo.items.values()

            loop.run_forever()
        except KeyboardInterrupt:
            logger.debug('Stopping the event loop by keyboard interrupt')
            loop.stop()
            os._exit(0)
        except Exception as e:
            logger.warn(repr(e))
            raise
        finally:
            logger.debug('Closing the event loop')
            loop.close()

        pass # if shouldQuery == True:



class GoogleFinanceDataCrawler(FinanceDataCrwaler):
    """일단 구현을 하지만 쓰지 않는 것이 좋겠다. 이유는 두 가지.
    1.kospi 정보를 못믿음.
    2.kosdaq 정보가 없음."""
    def __init__(self):
        warnings.warn("Deprecated: 일단 구현을 하지만 쓰지 않는 것이 좋겠다.", DeprecationWarning)
        return

    def makeCode(self, code, marketType):
        if marketType == MARKET_TYPE.KOSDAQ:
            code = 'KOSDAQ:%s' % code
        elif marketType == MARKET_TYPE.KOSPI:
            code = 'KRX:%s' % code
        return code

    def downloadStockPriceData(self, code, marketType, start_date, end_date):
        df = web.DataReader(self.makeCode(code,marketType),
                            "google",
                            start_date,
                            end_date)
        return df

    def updateAllStockPriceData(self,
                                stockDBCon,
                                priceDBCon,
                                stockInfo,
                                marketType,
                                start_date,
                                end_date,
                                row_unit='day'):
        msg = ''
        msg += '\n[%s] Start Downloading Stock Price Data' % getMethodName()
        msg += '    start date: %s' % str(start_date.date())
        msg += '    end date: %s' % str(end_date.date())
        logger.info(msg)

        totalCnt = len(stockInfo.items)
        curCnt = 0

        haveToQuery = True
        for i in stockInfo.items.values():
            shouldQuery = True

            curCnt += 1
            logger.debug( "[%d/%d][code:%s]" % \
                          (curCnt,
                             totalCnt,
                             self.makeCode(StockInfoItem.getCode(i), marketType)))

            if marketType.value != StockInfoItem.getMarketType(i):
                continue

            if StockInfoItem.getIsDelisted(i) == 1:
                continue

            qryStartDate = None
            qryEndDate   = None

            if StockInfoItem.getLastUpdate(i) == None:
                shouldQuery = True
                qryStartDate = start_date
                qryEndDate = end_date
                self.createTable(priceDBCon,
                                 makePriceTableName(StockInfoItem.getCode(i)))
            else:
                shouldQuery, qryStartDate, qryEndDate =\
                    self.maybeQueryItem(convertStringToDateTime(StockInfoItem.getStartDate(i)),
                                        convertStringToDateTime(StockInfoItem.getEndDate(i)),
                                        start_date,
                                        end_date)

            if shouldQuery == True:
                try:
                    df = self.downloadStockPriceData(StockInfoItem.getCode(i),
                                                     marketType,
                                                     qryStartDate,
                                                     qryEndDate)
                    df.to_sql(makePriceTableName(StockInfoItem.getCode(i)),
                              priceDBCon,
                              index_label='Date',
                              if_exists='append',
                              dtype={'Date': 'timestamp',
                                     'Open': 'integer',
                                     'High': 'integer',
                                     'Low': 'integer',
                                     'Close': 'integer',
                                     'Volume': 'integer'})
                    # todo: 성능적인 개선을 고려해보자.. del df을 하는 기준과 방법.
                    del df

                except Exception as e:
                    logger.warn("[code:%s][%s]" % (StockInfoItem.getCode(i), str(e)))
                else:
                    stockInfo.updateStockDate(stockDBCon,
                                              StockInfoItem.getCode(i),
                                              getCurrentTime(),
                                              qryStartDate,
                                              qryEndDate)
                pass
            pass # for i in stockInfo.items.values()
        logger.info("[%s] end downloading \n\n\n" % datetime.datetime.now() )

        pass

    pass

class YahooFinanceDataCrawler(FinanceDataCrwaler):
    """일단 구현을 하지만 쓰지 않는 것이 좋겠다. 이유는 두 가지.
    1.kospi 정보를 못믿음.
    2.kosdaq 정보가 없음."""
    def __init__(self):
        warnings.warn("Deprecated: 일단 구현을 하지만 쓰지 않는 것이 좋겠다.", DeprecationWarning)
        return

    def makeCode(self, code, marketType):
        if marketType == MARKET_TYPE.KOSDAQ:
            code += '.KQ'
        elif marketType == MARKET_TYPE.KOSPI:
            code += '.KS'
        return code

    def downloadStockPriceData(self, code, marketType, start_date, end_date):
        df = web.DataReader(self.makeCode(code,marketType),
                            "yahoo",
                            start_date,
                            end_date)
        return df

    def updateAllStockPriceData(self,
                                stockDBCon,
                                priceDBCon,
                                stockInfo,
                                marketType,
                                start_date,
                                end_date,
                                row_unit='day'):
        msg = ''
        msg += '\n[%s] Start Downloading Stock Price Data' % getMethodName()
        msg += '    start date: %s' % str(start_date.date())
        msg += '    end date: %s' % str(end_date.date())
        logger.info(msg)

        totalCnt = len(stockInfo.items)
        curCnt = 0

        haveToQuery = True
        for i in stockInfo.items.values():
            shouldQuery = True

            curCnt += 1
            logger.debug( "[%d/%d][code:%s]" % \
                          (curCnt,
                             totalCnt,
                             self.makeCode(StockInfoItem.getCode(i), marketType)))

            if marketType.value != StockInfoItem.getMarketType(i):
                continue

            if StockInfoItem.getIsDelisted(i) == 1:
                continue

            qryStartDate = None
            qryEndDate   = None

            if StockInfoItem.getLastUpdate(i) == None:
                shouldQuery = True
                qryStartDate = start_date
                qryEndDate = end_date
                self.createTable(priceDBCon,
                                 makePriceTableName(StockInfoItem.getCode(i)))
            else:
                shouldQuery, qryStartDate, qryEndDate =\
                    self.maybeQueryItem(convertStringToDateTime(StockInfoItem.getStartDate(i)),
                                        convertStringToDateTime(StockInfoItem.getEndDate(i)),
                                        start_date,
                                        end_date)

            if shouldQuery == True:
                try:
                    df = self.downloadStockPriceData(StockInfoItem.getCode(i),
                                                     marketType,
                                                     qryStartDate,
                                                     qryEndDate)
                    # 'adj close'는 없는 것이 나음.
                    del df['Adj Close']

                    df.to_sql(makePriceTableName(StockInfoItem.getCode(i)),
                              priceDBCon,
                              index_label='Date',
                              if_exists='append',
                              dtype={'Date': 'timestamp',
                                     'Open': 'integer',
                                     'High': 'integer',
                                     'Low': 'integer',
                                     'Close': 'integer',
                                     'Volume': 'integer'})
                    # todo: 성능적인 개선을 고려해보자.. del df을 하는 기준과 방법.
                    del df

                except Exception as e:
                    logger.warn("[code:%s][%s]" % (StockInfoItem.getCode(i), str(e)))
                else:
                    stockInfo.updateStockDate(stockDBCon,
                                              StockInfoItem.getCode(i),
                                              getCurrentTime(),
                                              qryStartDate,
                                              qryEndDate)
                pass
            pass # for i in stockInfo.items.values()
        logger.info("[%s] end downloading\n\n " % datetime.datetime.now() )

        pass
    pass



if __name__ == '__main__':
    import shutil

    DBManager.init()
    # DBManager.adjustDBPragma()

    stockCon = DBManager.getStockInfoDBConnection()
    priceCon = DBManager.getStockPriceDBConnection()

    # 아래 주석을 제거하면 18초에서 15초 정도로 성능 개선 있음.
    # 데이터가 많으면 더욱 클것으로 보임.

    # stockInfo = StockInfo()
    # stockInfo.readFromDatabase(stockCon, MARKET_TYPE.KOSPI)
    # st = datetime.datetime.now()

    # 네이버에서 한달분량 긁어오는데 3분 걸린다..
    # 증권사 API를 이용하는게 짱일듯 하다.
    # nvCrawler = NaverFinanceDataCrawler()
    # nvCrawler.updateAllStockPriceData(stockCon,
    #                                   priceCon,
    #                                   stockInfo,
    #                                   MARKET_TYPE.KOSPI,
    #                                   datetime.datetime(2010, 1, 1, 0, 0, 0),
    #                                   datetime.datetime(2016, 10, 30, 0, 0, 0) )

    stockInfo = StockInfo()
    stockInfo.loadFromDatabase(stockCon, MARKET_TYPE.KOSPI)
    st = datetime.datetime.now()

    # nvCrawler = NaverFinanceDataCrawler()
    # nvCrawler.updateAllStockPriceData(stockCon,
    #                                   priceCon,
    #                                   stockInfo,
    #                                   MARKET_TYPE.KOSDAQ,
    #                                   datetime.datetime(2010, 1, 1, 0, 0, 0),
    #                                   datetime.datetime(2016, 10, 30, 0, 0, 0) )

    # google이 야후에 비해 빠르고 정보도 많다.
    # 정확성은 모르겠음..
    googleCrawler = GoogleFinanceDataCrawler();
    googleCrawler.updateAllStockPriceData(stockCon,
                                          priceCon,
                                          stockInfo,
                                          MARKET_TYPE.KOSPI,
                                          datetime.datetime(2010,1,1,0,0,0),
                                          datetime.datetime(2016,10,30,0,0,0) )

    et = datetime.datetime.now()
    delta = et - st
    logger.info("Elapsed Time: [%d:%d]" % (int(delta.seconds / 60), (delta.seconds % 60)))

    DBManager.finalize()