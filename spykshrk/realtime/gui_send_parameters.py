from spykshrk.realtime.realtime_base import PrintableMessage

class TargetArmMessage(PrintableMessage):
    def __init__(self, value:int):
        self.value = value

class PosteriorThresholdMessage(PrintableMessage):
    def __init__(self, value:float):
        self.value = value

class NumAboveThreshMessage(PrintableMessage):
    def __init__(self, value:int):
        self.value = value

class PositionLimitMessage(PrintableMessage):
    def __init__(self, value):
        self.value = value

class MaxCenterWellDistMessage(PrintableMessage):
    def __init__(self, value:float):
        self.value = value

class RippleThresholdMessage(PrintableMessage):
    def __init__(self, value:float):
        self.value = value

class CondRippleThresholdMessage(PrintableMessage):
    def __init__(self, value:float):
        self.value = value

class VelocityThresholdMessage(PrintableMessage):
    def __init__(self, value:float):
        self.value = value

class ShortcutMessage(PrintableMessage):
    def __init__(self, value:bool):
        self.value = value

class InstructiveTaskMessage(PrintableMessage):
    def __init__(self, value:bool):
        self.value = value

class RippleCondOnlyMessage(PrintableMessage):
    def __init__(self, value:bool):
        self.value = value

