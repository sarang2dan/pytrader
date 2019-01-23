from pywinauto import application
from pywinauto import timings
import time
import os
import signal


# import Utils.ConfManager

def utStartKiwoomUpgrader():
# todo: 설정파일로 부터 키움 실행파일경로를 받아야 한다.
# todo: 사용자 암호와 공인인증서암호도 외부로부터 받자.
    app = application.Application()
    # todo: exception 처리 및 로그처리.
    app.start('C:/Kiwoom/KiwoomFlash2/khministarter.exe')

    title = '번개 Login'
    dlg = timings.WaitUntilPasses(20,
                                  0.5,
                                  lambda: app.window_(title=title))
    #todo:  ID 도 구현
    ctrlPwd = dlg.Edit2
    #ctrlPwd.setFocus()
    ctrlPwd.TypeKeys('')

    ctrlCert = dlg.Edit3
    #ctrlCert.setFocus()
    ctrlCert.TypeKeys('')

    ctrlOKBtn = dlg.Button0
    ctrlOKBtn.Click()

# todo: sleep time을 설정파일로 받아들이기
    sMaxWaitTime = 60
    for i in range(0, sMaxWaitTime):
        print("%d초후 종료 %d초" % (sMaxWaitTime, i+1) )
        time.sleep(1)
#todo: 깔끔하게 처리할 수 없나?
    os.system("taskkill /im khmini.exe")
    return