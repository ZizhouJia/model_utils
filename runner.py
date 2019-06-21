import multiprocessing
import time
import traceback
from copy import deepcopy

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetCount
from tensorboardX import SummaryWriter


class worker(multiprocessing.Process):
    def __init__(self, t):
        multiprocessing.Process.__init__(self)
        self.t = t

    def run(self):
        solver = self._init_task(self.t)
        solver.main_loop()

    def _init_task(self, t):
        solver = t.solver["class"](**t.solver["params"])
        solver.set_config(t.config)
        return solver


class task(object):
    def __init__(self):
        self.task_name = None
        self.solver = None
        self.config = None
        self.memory_use = []

# a runner is a controller of a set of tasks and a set of gpus


class runner(object):
    def __init__(self):
        self.writer = SummaryWriter("runs/runner-logs")
        nvmlInit()
        self.nvidia_free = []
        self.nvidia_total = []
        self.tasks = []
        self.running_tasks = []
        self.workers = []
        self.logic_free = []
        self.spawn = False

    def set_spawn(self):
        self.spawn = True

    def update_nvidia_info(self):
        nvidia_free = []
        nvidia_total = []
        deviceCount = nvmlDeviceGetCount()
        for i in range(0, deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            meminfo = nvmlDeviceGetMemoryInfo(handle)
            nvidia_free.append(meminfo.free/1024/1024)
            nvidia_total.append(meminfo.total/1024/1024)
        self.nvidia_free = nvidia_free
        self.nvidia_total = nvidia_total
        logic_free = nvidia_free[:]
        for i in range(0, len(self.running_tasks)):
            t = self.running_tasks[i]
            for j in range(0, len(t.config.device_use)):
                logic_free[t.config.device_use[j]] -= t.config.memory_use[j]
        self.logic_free = logic_free

    def generate_tasks(self, task_list):
        tasks = []
        for i in range(0, len(task_list)):
            t = task()
            t.task_name = task_list[i]
            t.task_name = t.task_name["config"].task_name
            t.solver = task_list[i]["solver"]
            t.config = deepcopy(task_list[i]["config"])
            t.memory_use = task_list[i]["config"].memory_use
            tasks.append(t)
        self.tasks = tasks

    def _dispatch_cards(self, memory_use):
        card_use = []
        for i in range(0, len(memory_use)):
            mem = memory_use[i]
            for j in range(0, len(self.nvidia_free)):
                if(j in card_use or mem > self.nvidia_free[j] or
                        mem > self.logic_free[j]):
                    continue
                else:
                    card_use.append(j)
                    break
        if(len(card_use) == len(memory_use)):
            return card_use
        else:
            return -1

    def _check_card(self, memory_use, device_use):
        if(len(memory_use) != len(device_use)):
            print("The mem use is not equal to deivce_use")
            return False
        for i in range(0, len(device_use)):
            if(self.nvidia_free[device_use[i]] < memory_use[i] or
                    self.logic_free[device_use[i]] < memory_use[i]):
                return False
        return True

    def main_loop(self):
        if(self.spawn):
            multiprocessing.set_start_method("spawn")
        while(len(self.tasks) != 0 or len(self.running_tasks) != 0):
            self.update_nvidia_info()
            handled = -1
            for i in range(0, len(self.tasks)):
                t = self.tasks[i]
                device_use = -1
                if(t.config.device_use is not None):
                    if(self._check_card(t.memory_use, t.config.device_use)):
                        device_use = t.config.device_use
                else:
                    device_use = self._dispatch_cards(t.memory_use)
                if(device_use != -1):
                    print("************************begin task " +
                          t.task_name+"****************")
                    t.config.device_use = device_use
                    w = worker(t)
                    w.start()
                    self.running_tasks.append(t)
                    self.workers.append(w)
                    handled = i
                    time.sleep(20)
                    break
            if(handled != -1):
                del self.tasks[handled]
            else:
                if(len(self.tasks) != 0):
                    print("no device can be used")
                time.sleep(10)

            end_index = -1
            for i in range(0, len(self.running_tasks)):
                t_name = self.running_tasks[i].task_name
                if(not self.workers[i].is_alive()):
                    end_index = i
                    print("************************end task " +
                          t_name+"****************")
                    break
            if(end_index != -1):
                del self.running_tasks[end_index]
                del self.workers[end_index]
