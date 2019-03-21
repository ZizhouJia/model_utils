from flask import Flask,request
from flask_apscheduler import APScheduler
from tensorboardX import SummaryWriter

class task_signal:
    self.WAIT_STOP=0
    self.STOP=1
    self.WAIT_RUNING=2
    self.RUNING=3
    self.WAIT=4
    self.CONNECT_OUT=5
    self.ERROR=6



class task_scheduler(obect):
    def __init__(self,port=6008):
        self.writer=SummaryWriter("runs/task_logs")
        self.tasks_state={}
        self.tasks_lasttime_connect={}
        self.tasks_list=[]
        self.current_wait_running=-1
