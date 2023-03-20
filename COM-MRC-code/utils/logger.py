import logging
import sys,time,os
now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))

class Logger:
    def __init__(self,filename,filepath='./logger'):

        os.makedirs(filepath, exist_ok=True)

        logger = logging.getLogger('')
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f'{filepath}/{filename}.log')
        sh = logging.StreamHandler(sys.stdout)
        # formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s [%(filename)s.%(funcName)s:%(lineno)d] %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(sh)

        self.logger = logger

