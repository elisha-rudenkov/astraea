import numpy as np

def angle_difference(new_angle, calibration_angle):
    """
    Computes the minimal difference between two angles (in degrees)
    while properly handling the wrap-around at ±180°.
    Returns a value in the range [-180, 180].
    """
    diff = new_angle - calibration_angle
    diff = (diff + 180) % 360 - 180
    return diff 