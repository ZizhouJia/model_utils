import os
import logging
import torch
import tensorboardX
import torchvision.utils as vutils
import cv2
import numpy as np


class writer(object):
    def __init__(self, task_name, summary_writer_default_path="runs",
                 logger_default_path="logs", image_default_path="images",output_log=True):
        # set defualt path
        self.output_log=output_log
        self._create_default_path(summary_writer_default_path)
        self._create_default_path(logger_default_path)
        self._create_default_path(image_default_path)
        self._summary_writer=None
        # summary writer init
        if(self.output_log):
            summary_writer_path = os.path.join(
                summary_writer_default_path, task_name)
            self._create_default_path(summary_writer_path)
            self._summary_writer = tensorboardX.SummaryWriter(summary_writer_path)
        # init logger
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)
        formater = logging.Formatter("%(process)d - %(asctime)s - %(message)s")
        # set file handler
        if(self.output_log):
            fh = logging.FileHandler(os.path.join(
                logger_default_path, task_name+".log"))
            fh.setLevel(logging.INFO)
            fh.setFormatter(formater)
            self._logger.addHandler(fh)
        # set screen handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formater)
        self._logger.addHandler(ch)

        # init image writer path
        self._image_writer_path = os.path.join(image_default_path, task_name)
        self._create_default_path(self._image_writer_path)

    def _create_default_path(self, path):
        if(not os.path.exists(path)):
            os.makedirs(path)

    def write_message(self,message):
        self._logger.info(message)

    def write_log(self, log_dict, epoch=-1, iteration=-1, step=-1):
        output_string = "("
        if(epoch != -1):
            output_string += str(epoch)
            output_string += ", "
        if(iteration != -1):
            output_string += str(iteration)
            output_string += ", "
        if(step != -1):
            output_string += str(step)
            output_string += ", "
        if(output_string[-2] == ","):
            output_string = output_string[:-2]
        output_string += "): "
        self.write_message(output_string+str(log_dict))

    def write_board_line(self, log_dict, index):
        if(self._summary_writer is None):
            self._logger.warning("The summary writer closed")
            return
        for key in log_dict:
            self._summary_writer.add_scalar("scalar/"+key, log_dict[key])

    def write_board_image(self, image_dict, index):
        if(self._summary_writer is None):
            self._logger.warning("The summary writer closed")
            return
        for key in image_dict:
            self._summary_writer.add_image(
                'image/'+key, vutils.make_grid(image_dict[key], 1), index)

    def write_file_image(self, images, key_string):
        index_path = os.path.join(self._image_writer_path, str(key_string))
        self._create_default_path(index_path)

        if(isinstance(images, dict)):
            for k in images:
                images[k] = cv2.cvtColor(images[k], cv2.COLOR_M_RGBA2BGR)
                cv2.imwrite(os.path.join(index_path, str(k)+".jpg"), images[k])
            return

        if(isinstance(images, list) or isinstance(images, np.ndarray)):
            for k in range(0, len(images)):
                images[k] = cv2.cvtColor(images[k], cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(index_path, str(k)+".jpg"), images[k])
            return

        print("unsupport image save data type")
 
