import numbers
import numpy as np


def pretty_str_type(s):
    if type(s) is str:
        return '\'' + s + '\''
    elif isinstance(s, numbers.Number):
        return str(s)


def pretty_str_list(en, buffer=3):
    if type(en) is list and len(en) > buffer*2:
        return '[' + ', '.join([pretty_str_type(i) for i in en[0:buffer]]) + ', ... , ' + \
               ', '.join([pretty_str_type(i) for i in en[len(en)-buffer-1:-1]]) + ']'
    elif type(en) is np.ndarray and len(en) > buffer*2:
        return 'array([' + ', '.join(en[0:buffer].astype('str')) + ', ... , ' + \
               ', '.join(en[len(en)-buffer-1:-1].astype(str)) + '])'
    else:
        return en.__str__()