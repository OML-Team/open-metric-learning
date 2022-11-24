class BaseOMLException(Exception):
    pass


class InvalidBBoxesException(BaseOMLException):
    pass


class InferenceConfigError(BaseOMLException):
    pass
