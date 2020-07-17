import warnings
import sys


# Monkey patching show warning
def _showwarning(message, category, filename, lineno, file=None, line=None):
    if file is None:
        file = sys.stderr
    print('{}:{}: {}:\n{}'.format(filename, lineno, category.__name__, message), file=file)

warnings.showwarning = _showwarning


class FrankLabDeprecationWarning(Warning):
    pass


class ConstructorWarning(UserWarning):
    pass


class OverridingAttributeWarning(UserWarning):
    pass

