class ImageLoadError(Exception):
    def __init__(self, message: str = 'Failed to load image from file'):
        self.message = f'{message}'
        super().__init__(self.filename)