'''
#! /usr/bin/python3

import aiohttp
import asyncio
import os.path
import re
import sys
import json
import sqlite3
from bs4 import BeautifulSoup
from collections import namedtuple
from datetime import datetime, date
import logger
from os import getpid

def main(date_from, date_to, symbols):
    logger.info("Starting process (pid:{})".format(getpid()))
    logger.info("This pricess will fetch interday prices from NAVER")

    global symbols_name
    global semaphore
    global conn
    global cursor
    global inserted_rows

    inserted_rows = 0
    DB = 'interday.db'

    try:
        logger.info('Loading symbols description file')

        symbols_name = {}
        for symbol, values in json.load(open('symbols.txt', 'r')).items():
            symbols_name[symbol] = values[0]

    except FileNotFoundError:
        logger.error('symbols.txt has to be prepared if symbols were not passed as arguments')
        raise

    try:
        logger.info('Connecting to database, {}'.format(DB))

        conn = sqlite3.connect(DB)
        cursor = conn.cursor()
        cursor.execute("PRAGMA synchronous = OFF")
        cursor.execute("PRAGMA journal_mode = OFF")

        row = cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='interday'").fetchone()
        if row is None:
            logger.info('Creating a table')

            cursor.execute("""CREATE TABLE interday (
                                symbol TEXT,
                                date DATE,
                                open INTEGER,
                                high INTEGER,
                                low INTEGER,
                                close INTEGER,
                                volume INTEGER)""")

            cursor.execute("""CREATE UNIQUE INDEX interday_idx
                                ON interday (symbol, date)""")
    except sqlite3.Error:
        logger.error('Database error occurred')
        raise

    try:
        logger.debug('Preparing an event loop')

        loop = asyncio.get_event_loop()
        semaphore = asyncio.Semaphore(10)

        # for testing
        # date_from = datetime.strptime('2016-09-01', '%Y-%m-%d').date() # for testing
        # symbols = ('015760', '005820') # for testing

        if not symbols:
            symbols = tuple(symbols_name.keys())

        logger.debug('Registering initial {} tasks for event loop'.format(len(symbols)))
        [loop.create_task(fetch(symbol, date_from, date_to)) for symbol in symbols]

        loop.run_forever()

    except KeyboardInterrupt:
        logger.debug('Stopping the event loop by keyboard interrupt')
        loop.stop()

    except Exception as e:
        logger.warn(repr(e))
        raise

    finally:
        logger.debug('Closing the event loop')
        loop.close()

        logger.debug('{} rows were inserted into database'.format(inserted_rows))
        logger.debug('Closing the database connection')
        conn.close()


async def fetch(symbol, date_from, date_to, page=1):
    global semaphore

    with await semaphore:
        text = await request(symbol, page)
        prices = await parse(symbol, date_from, date_to, page, text)
        await save(symbol, prices)
        await asyncio.sleep(0.5)

    if len(asyncio.Task.all_tasks()) == 1:
        asyncio.get_event_loop().stop()


async def request(symbol, page):
    url = 'http://finance.naver.com/item/sise_day.nhn?code={}&page={}'.format(symbol, page)
    logger.debug("Requesting {} ({})".format(url, symbols_name[symbol]))

    TIMEOUT = 4

    try:
        with aiohttp.Timeout(TIMEOUT):
            async with aiohttp.request('GET', url) as resp:
                assert resp.status == 200
                return await resp.text(encoding='euc-kr')

    except asyncio.TimeoutError:
        logger.warn("Timeout {} seconds from requesting".format(TIMEOUT))
        asyncio.get_event_loop().create_task(fetch(symbol, page))
        return None


async def parse(symbol, date_from, date_to, page, text):
    logger.debug(
        "Parsing {:,d} bytes for {}({}) on page {}".format(len(text), symbols_name[symbol], symbol, page))

    Price = namedtuple('Price', ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume'])
    prices = []
    lastpage = 1

    try:
        bs = BeautifulSoup(text, 'html.parser')

        for tr in bs.find_all('tr'):
            if tr.span == None:
                continue

            tds = tr.find_all('td')

            date = datetime.strptime(tds[0].text, '%Y.%m.%d').date()
            if date > date_to:
                continue
            if date < date_from:
                return prices

            close = int(tds[1].text.replace(',', ''))
            gap = int(tds[2].text.replace(',', ''))  # unused
            open = int(tds[3].text.replace(',', ''))
            high = int(tds[4].text.replace(',', ''))
            low = int(tds[5].text.replace(',', ''))
            volume = int(tds[6].text.replace(',', ''))

            prices.append(Price(symbol, date, open, high, low, close, volume))

        link_lastpage = bs.find_all("td", class_="pgRR")[0].a.get('href')
        lastpage = int(re.search(r'page=(\d+)', link_lastpage).group(1))

    except (ValueError, IndexError):
        logger.warn("Parsing error: symbol {} (page: {})".format(symbol, page))
        # logger.warn(text)   # for debugging

    if page < lastpage:
        asyncio.get_event_loop().create_task(fetch(symbol, date_from, date_to, page + 1))

    return prices


async def save(symbol, prices):
    logger.debug("Saving prices for {:,d} days of {}({})".format(len(prices), symbols_name[symbol], symbol))

    global conn
    global cursor
    global inserted_rows

    for p in prices:
        try:
            cursor.execute("""INSERT INTO interday (symbol, date, open, high, low, close, volume) VALUES
                ('{p.symbol}', '{p.date:%Y-%m-%d}', {p.open}, {p.high}, {p.low}, {p.close}, {p.volume})""".format(p=p))
            inserted_rows += cursor.rowcount

        except sqlite3.IntegrityError:
            cursor.execute("""UPDATE interday SET
                open = {p.open}, high = {p.high}, low = {p.low}, close = {p.close}, volume = {p.volume}
                WHERE symbol = '{p.symbol}' AND date = '{p.date:%Y-%m-%d}'""".format(p=p))

    conn.commit()


if __name__ == '__main__':
    date_from = date.today()
    date_to = date.today()
    symbols = None

    try:
        if len(sys.argv) >= 2:
            date_from = datetime.strptime(sys.argv[1], '%Y-%m-%d').date()

        if len(sys.argv) >= 3:
            date_to = datetime.strptime(sys.argv[2], '%Y-%m-%d').date()

        if len(sys.argv) >= 4:
            symbols = sys.argv[3:]

    except:
        print("usage: {} [2016-01-01 [2016-12-31 [005930 [015760 ...]]]".format(sys.argv[0]))
        raise

    main(date_from, date_to, symbols)
'''