import numpy as np

np.set_printoptions(precision=4, suppress=True)

def get_line_by_two_points(p1, p2):
    print("line from two points")
    x1 = np.array([p1[0], p1[1], 1])
    x2 = np.array([p2[0], p2[1], 1])
    l = np.cross(x1, x2)
    print("x1 : ", x1)
    print("x2 : ", x2)
    l = l / np.sqrt(np.sum(l[:2]**2))  # normalize the xy
    print(" l : ", l)
    return l

p11 = [42, 142]
p12 = [218, 13]
line1 = get_line_by_two_points(p11, p12)
print("line1 : ", line1)

p21 = [221, 218]
p22 = [23, 268]
line2 = get_line_by_two_points(p21, p22)
print("line2 : ", line2)

def get_point_by_two_lines(l1, l2):
    print("point from two lines")
    l1 = np.array(l1)
    l2 = np.array(l2)
    print("l1 : ", l1)
    print("l2 : ", l2)
    p = np.cross(l1, l2)
    print("p : ", p)
    p = p / p[2]
    print("p : ", p)
    return p


# three points, both lines contain x1
p1 = [218, 16]
p2 = [42, 141]
p3 = [347, 94]

l1 = get_line_by_two_points(p1, p2)
l2 = get_line_by_two_points(p1, p3)
point1 = get_point_by_two_lines(l1, l2)

print("out :", point1)


def get_horizon_by_two_parallel_groundlines(l1, l2):
    pass

def get_vertical_center_line_by_two_parallel_vertical_lines(l1, l2):
    pass

def get_intrinsics_by_two_ground_lines_and_vertical_center(l1, l2, c):
    pass

