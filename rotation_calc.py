import numpy as np
np.set_printoptions(precision=7, suppress=True)

def cosd(x):
    return np.cos(np.radians(x))

def sind(x):
    return np.sin(np.radians(x))

def create_zyx_rot(att):
    """param : [yaw, pitch, roll]\n
    return : numpy rotation matrix"""
    c1, s1 = cosd(att[0]), sind(att[0])
    c2, s2 = cosd(att[1]), sind(att[1])
    c3, s3 = cosd(att[2]), sind(att[2])
    rot_mtx = np.array([[c1*c2, c1*s2*s3-s1*c3, c1*c3*s2+s1*s3]
                        ,[s1*c2, s1*s2*s3+c1*c3, s1*s2*c3-c1*s3]
                        ,[-s2, c2*s3, c2*c3]])
    return rot_mtx

def get_cam_to_body_mtx(cam_att):
    out = create_zyx_rot(cam_att)
    return out.T

att = [-90, 0, 0]

rmat = create_zyx_rot(att)
print("rmat :\n", rmat)

cam_rot = np.zeros((3,3), dtype=np.float32)
cam_rot[0,1] = 1.0
cam_rot[1,0] = -1.0
cam_rot[2,2] = 1.0
print("cam_rot :\n", cam_rot)



comb = np.dot(cam_rot, rmat)
print("comb :\n", comb)

a = np.array([1,1,1])


b = np.array([1,1,1])

print(a-b)
