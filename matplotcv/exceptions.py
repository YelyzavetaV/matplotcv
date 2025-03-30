class PipelineError(Exception):

    def __init__(self, message: str):
        self.message = f'{message}'
        super().__init__(self.filename)
