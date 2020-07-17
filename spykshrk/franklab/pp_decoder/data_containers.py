import warnings
from spykshrk.franklab.warnings import FrankLabDeprecationWarning
from spykshrk.franklab.data_containers import *


warnings.warn(FrankLabDeprecationWarning('Importing from {} deprecated, please import from '
                                         'spykshrk.franklab.data_containers.'.format(__name__)))

