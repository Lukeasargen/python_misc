import numpy as np



def rotate_2d(vector, angle, degrees = False):
    """
    Rotate a 2d vector counter-clockwise by @angle.\n
    vector  : [x, y]\n
    angle   : rotation angle
    degrees : set True if angle is in degrees, default is False. 
    """
    if degrees:
        angle = np.radians(angle)
    v = np.array(vector[:2])
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    return R.dot(v)


v = [1, 0]
a = 63

out = rotate_2d(v, a, degrees=True)

print("out :", out)

