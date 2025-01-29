import math
from collections import deque
import numpy as np

#1 part descriptor is a dimension
class DimensionDescriptor:
    def __init__(self, type: str, name = "", expected = 0.0, min = 0.0, max = 0.0, contourArcLen = 0.0, contourArea = 0.0, widthRefLen = 0, heightRefLen = 0, refRadius = 0):
        self.type = type
        self.name = name
        self.expected = expected
        self.min = min
        self.max = max
        self.contourArcLen = contourArcLen
        self.contourArea = contourArea
        self.widthRefLen = widthRefLen
        self.heightRefLen = heightRefLen

        self.refRadius = refRadius
        self.circleParams = None

        self.centerHistory = deque(maxlen=20)  # number of frames to average the circle
        self.radiusHistory = deque(maxlen=20)

    def __repr__(self):
        return (f"MyClass(type='{self.type}', name='{self.name}', "
                f"expected={self.expected}, min={self.min}, max={self.max}, contourArcLen={self.contourArcLen}, contourArea={self.contourArea})"
                f"heightrefLen={self.heightRefLen}, widthRefLen={self.widthRefLen}")

    def contains(self, partDescriptor, type):
        return any(descriptor.type == type for descriptor in partDescriptor)

    def updateAvgCircle(self, center, radius):
        self.centerHistory.append(center)
        self.radiusHistory.append(radius)

    def getAvgCircle(self):
        if len(self.centerHistory) <= 10:
            return 0,0
        avg_center = np.mean(self.centerHistory, axis=0)
        avg_radius = np.mean(self.radiusHistory)
        return avg_center, avg_radius

    def findClosestSide(self, sides):
        closestWidthSide = None
        closestHeightSide = None
        closestWidthOpposite = None
        closestHeightOpposite = None
        minWidthDiff = float('inf')
        minHeightDiff = float('inf')
        widthIndex = heightIndex = -1

        # Step 1: Find the closest width and height sides
        for i, side in enumerate(sides):
            length = math.dist(side[0], side[1])

            # Compare with saved width length
            if self.widthRefLen is not None:
                widthDiff = abs(length - self.widthRefLen)
                if widthDiff < minWidthDiff:
                    minWidthDiff = widthDiff
                    closestWidthSide = side
                    widthIndex = i

            # Compare with saved height length
            if self.heightRefLen is not None:
                heightDiff = abs(length - self.heightRefLen)
                if heightDiff < minHeightDiff:
                    minHeightDiff = heightDiff
                    closestHeightSide = side
                    heightIndex = i

        # Step 2: Determine opposite sides based on index
        if widthIndex != -1:
            # Assuming opposite of sides[i] is sides[(i + 2) % len(sides)]
            oppositeIndex = (widthIndex + 2) % len(sides)
            closestWidthOpposite = sides[oppositeIndex]

        if heightIndex != -1:
            # Assuming opposite of sides[i] is sides[(i + 2) % len(sides)]
            oppositeIndex = (heightIndex + 2) % len(sides)
            closestHeightOpposite = sides[oppositeIndex]

        return closestWidthSide, closestWidthOpposite, closestHeightSide, closestHeightOpposite



