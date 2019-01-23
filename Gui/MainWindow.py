import sys
from PyQt5 import uic
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QAxContainer import *

from Utils.TradingFirmWrapper.Kiwoom import *

form_class = uic.loadUiType("Gui/resource/pytrader.ui")[0]

'''
  <widget class="QPushButton" name="psButtonSendOrder">
  <widget class="QComboBox" name="cbBoxNomialPriceType">
  <widget class="QLineEdit" name="lnEditStockCode">
  <widget class="QLineEdit" name="lnEditStockName">
  <widget class="QSpinBox" name="spBoxOrderQuantity">
  <widget class="QComboBox" name="cbBoxOrderType">
  <widget class="QComboBox" name="cbBoxAccounts">
  <widget class="QSpinBox" name="spBoxOrderPrice">
'''

class MainWindow(QMainWindow, form_class):
    orderType = {'신규매수': 1,
                 '신규매도': 2,
                 '매수취소': 3,
                 '매도취소': 4}
    # 호가
    nomialPriceType = {'지정가': "00",
                       '시장가': "03"}

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.kiwoom = Kiwoom()
        self.kiwoom.CommConnect()

        self.timer = QTimer(self)
        self.timer.start(1000)
        self.timer.timeout.connect(self.timeout)

        self.lnEditStockCode.textChanged.connect(self.onStockCodeChanged)
        self.psButtonSendOrder.clicked.connect(self.sendOrder)
        self.getAccountNumber()

        return

    def timeout(self):
        current_time = QTime.currentTime()
        text_time = current_time.toString("hh:mm:ss")
        time_msg = "현재시간: " + text_time

        state = self.kiwoom.GetConnectState()
        if state == 1:
            state_msg = "서버와 연결 되었음"
        else:
            state_msg = "서버와 접속 끊김"

        self.statusbar.showMessage(state_msg + " | " + time_msg)
        return

    def onStockCodeChanged(self):
        stockCode = self.lnEditStockCode.text()
        stockName = self.kiwoom.GetMasterCodeName(stockCode)

        self.lnEditStockName.setText(stockName)
        return

    def getAccountNumber(self):
        accountsCnt = int(self.kiwoom.GetLoginInfo("ACCOUNT_CNT"))
        accounts = self.kiwoom.GetLoginInfo("ACCNO")
        accountsList = accounts.split(';')[0:accountsCnt]
        self.cbBoxAccounts.addItems(accountsList)
        return

    def sendOrder(self):
        account = self.cbBoxAccounts.currentText()
        orderType = self.cbBoxOrderType.currentText()
        stockCode = self.lnEditStockCode.text()
        nomialPriceTp = self.cbBoxNomialPriceType.currentText()

        orderQuantity = self.spBoxOrderQuantity.value()
        orderPrice    = self.spBoxOrderPrice.value()

        print(account)
        print(orderType)
        print(stockCode)
        print(nomialPriceTp)
        print(orderQuantity)
        print(orderPrice)
        '''
        self.kiwoom.SendOrder("SendOrder_req",
                              "0101",
                              account,
                              self.orderType[orderType],
                              stockCode,
                              orderQuantity,
                              orderPrice,
                              self.nomialPriceType[nomialPriceTp],
                              "") #sOrgNum
        '''
        return
    pass