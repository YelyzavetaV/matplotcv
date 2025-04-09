from functools import wraps
from components import ErrorPopup


class PipelineError(Exception):

    def __init__(self, message: str):
        self.message = f'{message}'
        super().__init__(self.message)


def pipeline_error_handler(func):

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except PipelineError as e:
            error_popup = ErrorPopup()
            error_popup.message = str(e)
            error_popup.open()

    return wrapper
