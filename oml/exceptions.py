class BaseOMLException(Exception):
    pass


class InvalidBBoxesException(BaseOMLException):
    pass


class InferenceConfigError(BaseOMLException):
    pass


class InvalidDataFrameColumnsException(BaseOMLException):
    pass
