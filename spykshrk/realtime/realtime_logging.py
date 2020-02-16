import logging
import os


class MakeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=0):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logging.FileHandler.__init__(self, filename, mode, encoding, delay)


class LoggingClass(object):
    def __init__(self, *args, **kwds):
        super().__init__()
        self.class_log = logging.getLogger(name='{}.{}'.format(self.__class__.__module__,
                                                               self.__class__.__name__))


class PrintableMessage:

    def __str__(self):
        return '{:}({:})'.format(self.__class__.__name__, self.__dict__)
