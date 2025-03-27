# Abstract base class with common methods

class BaseVisualizer:
    def __init__(self):
        pass

    def visualize(self):
        raise NotImplementedError("Subclasses should implement this method.")
