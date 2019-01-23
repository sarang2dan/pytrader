import sqlite3
import os
import pandas

# todo: 싱글톤으로 구현해야함..
'''
class SingletonInstane:
  __instance = None

  @classmethod
  def __getInstance(DBManager):
    return DBManager.__instance

  @classmethod
  def instance(DBManager, *args, **kargs):
    DBManager.__instance = DBManager(*args, **kargs)
    DBManager.instance = DBManager.__getInstance
    return DBManager.__instance

class MyClass(BaseClass, SingletonInstane):
  pass

c = MyClass.instance()
'''

# todo:  일단 sqlite3으로 개발, InfluxDB를 이용하는 것으로 개발해봅시다.
class DBManager:
    stockInfoDB = None
    stockPriceDB = None
    # todo: 그외 수학모델과 머신러닝 결과값 저장하는 DB를 차용?

    dbPath = None

    def init():
        if DBManager.stockInfoDB == None and DBManager.stockPriceDB == None:
            DBManager.dbPath = os.getcwd() + os.path.sep + 'database'

            if os.path.isfile(DBManager.dbPath) == True:
                raise NotADirectoryError

            if os.path.isdir(DBManager.dbPath) == False:
                os.mkdir(DBManager.dbPath)

            # todo: 성능높이기위한 인자들이 보인다. cached_item 같은 것.. 나중에 고려해보기
            DBManager.stockInfoDB = sqlite3.connect(DBManager.dbPath + os.path.sep + 'stockInfo.db')
            DBManager.stockPriceDB = sqlite3.connect(DBManager.dbPath + os.path.sep + 'stockPrice.db')
            pass
        else:
            print('DB files has been already opened!!')

        return

    def finalize():
        if DBManager.stockInfoDB != None:
            DBManager.stockInfoDB.close()
            DBManager.stockInfoDB = None

        if DBManager.stockPriceDB != None:
            DBManager.stockPriceDB.close()
            DBManager.stockPriceDB = None

        return

    def getStockInfoDBConnection():
        if DBManager.stockInfoDB == None:
            raise sqlite3.DatabaseError
        return DBManager.stockInfoDB

    def getStockPriceDBConnection():
        if DBManager.stockPriceDB == None:
            raise sqlite3.DatabaseError
        return DBManager.stockPriceDB

    def adjustDBPragma():
        DBManager.stockPriceDB.execute("PRAGMA synchronous = OFF")
        DBManager.stockPriceDB.execute("PRAGMA journal_mode = OFF")
        DBManager.stockPriceDB.commit()
        return

    def resetDefaultDBPragma():
        DBManager.stockPriceDB.execute("PRAGMA synchronous = FULL")
        DBManager.stockPriceDB.execute("PRAGMA journal_mode = DELETE")
        return

    def loadStockPriceToDataFrame(table_name, start_date, end_date):
        sql = "select * from %s where Date between '%s' and '%s' " % \
              (table_name, str(start_date), str(end_date))
        df = pandas.read_sql(sql,
                             DBManager.getStockPriceDBConnection(),
                             index_col='Date')
        return df

if __name__ == '__main__':
    DBManager.init()
    DBManager.getStockInfoDBConnection()
    DBManager.finalize()
