# todo: 모듈의 초기화와 실행 순서 제어

from Gui.MainWindow import MainWindow
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QAxContainer import *
from Utils.ProgramUpgrader import utStartKiwoomUpgrader
import sys

if __name__ == "__main__":
    #todo: upgrader 정상동작확인
    # utStartKiwoomUpgrader()
    app = QApplication(sys.argv)
    myWindow = MainWindow()
    myWindow.show()
    app.exec_()

    del myWindow