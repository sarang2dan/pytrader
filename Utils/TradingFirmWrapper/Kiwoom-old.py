import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QAxContainer import QAxWidget
import time
import pandas as pd
import sqlite3

class Kiwoom(QAxWidget):
    def __init__(self):
        super().__init__()
        self.setControl("KHOPENAPI.KHOpenAPICtrl.1")

        self.connect(self,
                     SIGNAL("OnEventConnect(int)"),
                     self.OnEventConnect)

        self.connect(self,
                     SIGNAL("OnReceiveTrData(QString, QString, QString, QString, QString, int, QString, QString, QString)"),
                     self.OnReceiveTrData)

        self.connect(self,
                     SIGNAL("OnReceiveChejanData(QString, int, QString)"),
                     self.OnReceiveChejanData)
        return

    def CommConnect(self):
        self.login_event_loop = QEventLoop()
        self.dynamicCall("CommConnect()")
        self.login_event_loop.exec_()
        return

    def OnEventConnect(self, errCode):
        if errCode == 0:
            print("connected")
        else:
            print("disconnected")
        self.login_event_loop.exit()

        return

    def SetInputValue(self, sID, sValue):
        self.dynamicCall("SetInputValue(QString, QString)",
                         sID,
                         sValue)
        return

    def CommRqData(self,
                   aRQName,
                   aTRCode,
                   aPrevNext,
                   aScreenNo):
        self.tr_event_loop = QEventLoop()
        self.dynamicCall("CommRqData(QString, QString, int, QString)",
                         aRQName,
                         aTRCode,
                         aPrevNext,
                         aScreenNo)
        self.tr_event_loop.exec_()
        return

    def CommGetData(self,
                    aStockCode,
                    aRealType,
                    aFieldName,
                    aIndex,
                    aInnerFiledName):
        data = self.dynamicCall("CommGetDatadata(QString, QString, QString, int, QString)",
                                    aStockCode,
                                    aRealType,
                                    aFieldName,
                                    aIndex,
                                    aInnerFiledName)
        return data.strip()

    def OnReceiveChejanData(self, sGubun, nItemCnt, sFidList):
        print("sGubun: ", sGubun)
        print(self.GetChejanData(9203)) # 주문번호
        print(self.GetChejanData(302))  # 종목명
        print(self.GetChejanData(900))  # 주문수량
        print(self.GetChejanData(901))  # 가격
        print(self.GetChejanData(911))  # 체결량
        print(self.GetChejanData(910))  # 체결가
        '''
        #FID 설명
        #9201 계좌번호
        #9203 주문번호
        #9205 관리자사번
        #9001 종목코드, 업종코드
        #912 주문업무분류(JJ:주식주문, FJ:선물옵션, JG:주식잔고, FG:선물옵션잔고)
        #913 주문상태(10:원주문, 11:정정주문, 12:취소주문, 20:주문확인, 21:정정확인, 22:취소확인, 90 - 92:주문거부)
        #302 종목명
        #900 주문수량
        #901 주문가격
        #902 미체결수량
        #903 체결누계금액
        #904 원주문번호
        #905 주문구분(+현금내수, -현금매도…)
        #906 매매구분(보통, 시장가…)
        #907 매도수구분(1:매도, 2:매수)
        #908 주문 / 체결시간(HHMMSSMS)
        #909 체결번호
        #910 체결가
        #911 체결량
        #10  현재가, 체결가, 실시간종가
        #27  (최우선) 매도호가
        #28  (최우선) 매수호가
        #914 단위체결가
        #915 단위체결량
        #938 당일매매 수수료
        #939 당일매매세금
        '''
        return

    def GetLoginInfo(self, sTag):
        cmd = 'GetLoginInfo("%s")' % sTag
        ret = self.dynamicCall(cmd)
        return ret

    def OnReceiveTrData(self,
                        aScrNo,
                        aRQName,
                        aTrCode,
                        aRecordName,
                        aPrevNext,
                        aDataLength,
                        aErrorCode,
                        aMessage,
                        aSplmMsg):
        self.prev_next = aPrevNext

        if aRQName == "opt10081_req":
            cnt = self.GetRepeatCnt(aTrCode, aRQName)

            for i in range(cnt):
                date = self.CommGetData(aTrCode, "", aRQName, i, "일자")
                open = self.CommGetData(aTrCode, "", aRQName, i, "시가")
                high = self.CommGetData(aTrCode, "", aRQName, i, "고가")
                low  = self.CommGetData(aTrCode, "", aRQName, i, "저가")
                close  = self.CommGetData(aTrCode, "", aRQName, i, "현재가")

                self.ohlc['date'].append(date)
                self.ohlc['open'].append(int(open))
                self.ohlc['high'].append(int(high))
                self.ohlc['low'].append(int(low))
                self.ohlc['close'].append(int(close))

        while True:
            if self.tr_event_loop.isRunning():
                self.tr_event_loop.exit()
                break
            pass
        return

    def GetRepeatCnt(self, aTrCode, aRecordName):
        return self.dynamicCall("GetRepeatCnt(QString, QString)",
                                aTrCode,
                                aRecordName)

    def SendOrder(self,
                  sRQName,
                  sScreenNo,
                  sAccNo,
                  nOrderType,
                  sCode,
                  nQty,
                  nPrice,
                  sHogaGb,
                  sOrgOrderNo):
        self.dynamicCall("SendOrder(QString, QString, QString, int, QString, int, int, QString, QString)",
                         [sRQName, sScreenNo, sAccNo, nOrderType, sCode, nQty, nPrice, sHogaGb, sOrgOrderNo])
        return

    def GetChejanData(self, nFid):
        cmd = 'GetChejanData("%s")' % nFid
        ret = self.dynamicCall(cmd)
        return ret

    def GetCodeListByMarket(self, sMarket):
        cmd = 'GetCodeListByMarket("%s")' % sMarket
        ret = self.dynamicCall(cmd)
        item_codes = ret.split(';')
        return item_codes

    def GetMasterCodeName(self, aStrCode):
        cmd = 'GetMasterCodeName("%s")' % aStrCode
        ret = self.dynamicCall(cmd)
        return ret

    def GetConnectState(self):
        ret = self.dynamicCall("GetConnectState()")
        return ret

    def InitOHLCRawData(self):
        self.ohlc = {'date': [], 'open': [], 'high': [], 'low': [], 'close': []}
        return

'''
if __name__ == "__main__":
    app = QApplication(sys.argv)

    kiwoom = Kiwoom()
    kiwoom.CommConnect()
    kiwoom.InitOHLCRawData()

    # TR
    kiwoom.SetInputValue("종목코드", "039490")
    kiwoom.SetInputValue("기준일자", "20160624")
    kiwoom.SetInputValue("수정주가구분", 1)
    kiwoom.CommRqData("opt10081_req", "opt10081", 0, "0101")

    while kiwoom.prev_next == '2':
        time.sleep(0.2)
        kiwoom.SetInputValue("종목코드", "039490")
        kiwoom.SetInputValue("기준일자", "20160624")
        kiwoom.SetInputValue("수정주가구분", 1)
        kiwoom.CommRqData("opt10081_req", "opt10081", 2, "0101")

    df = pd.DataFrame(kiwoom.ohlc, columns=['open', 'high', 'low', 'close'], index=kiwoom.ohlc['date'])
    print(df.head())
    con = sqlite3.connect("c:/Users/Jason/stock.db")
    df.to_sql('039490', con, if_exists='replace')
'''