from math import atan, cos, sin, tan, radians, degrees


def calc_fov(fov, angle='dfov'):
    dfov, hfov, vfov = 0, 0, 0

    if angle == 'dfov':
        dfov = fov
        hfov = 2 * degrees(atan(tan(radians(dfov / 2)) *
                                cos(atan(img_height / img_width))))
        vfov = 2 * degrees(atan(tan(radians(dfov / 2)) *
                                sin(atan(img_height / img_width))))

    if angle == 'hfov':
        hfov = fov
        dfov = 2 * degrees(atan(tan(radians(hfov / 2)) /
                                cos(atan(img_height / img_width))))
        vfov = 2 * degrees(atan(tan(radians(dfov / 2)) *
                                sin(atan(img_height / img_width))))

    if angle == 'vfov':
        vfov = fov
        dfov = 2 * degrees(atan(tan(radians(vfov / 2)) /
                                sin(atan(img_height / img_width))))
        hfov = 2 * degrees(atan(tan(radians(dfov / 2)) *
                                cos(atan(img_height / img_width))))

    return dfov, hfov, vfov


# Test

img_width = 2592
img_height = 1944

# vfov = 48.8  # vertical field of view in degrees
# hfov = 62.2
# dfov = 74.09

fov = 56
angle = 'hfov'

out_dfov, out_hfov, out_vfov = calc_fov(fov, angle)

print("DFOV : {}\nHFOV : {}\nVFOV : {}".format(out_dfov, out_hfov, out_vfov))
