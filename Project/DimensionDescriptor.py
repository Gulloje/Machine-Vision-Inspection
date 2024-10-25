
#1 part descriptor is a dimension
class DimensionDescriptor:
    def __init__(self, type: str, name: str, expected: float, min: float, max: float, contourArcLen: float, contourArea: float):
        self.type = type
        self.name = name
        self.expected = expected
        self.min = min
        self.max = max
        self.contourArcLen = contourArcLen
        self.contourArea = contourArea

    def __repr__(self):
        return (f"MyClass(type='{self.type}', name='{self.name}', "
                f"expected={self.expected}, min={self.min}, max={self.max}, contourArcLen={self.contourArcLen}, contourArea={self.contourArea})")



