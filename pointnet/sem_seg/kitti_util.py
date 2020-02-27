import errno
import glob
import os

import numpy as np
import pcl

path_label = '/home/scosar/Documents/Datasets/KITTI/for_voxelnet/training/label_2/'
path_images = '/home/scosar/Documents/Datasets/KITTI/for_voxelnet/training/image_2/'
path_lidar = '/home/scosar/Documents/Datasets/KITTI/for_voxelnet/training/velodyne/'
calib_folder = '/home/scosar/Documents/Datasets/KITTI/data_object_calib/training/calib/'

labelList = sorted(glob.glob(os.path.join(path_label, "*.txt")))
labelImages = sorted(glob.glob(os.path.join(path_images, "*.png")))

CAM = 2
img_size = [370, 1224]

print(len(labelList))
print(len(labelImages))

all_labels = ['background', 'pedestrian', 'dontcare', 'car', 'cyclist', 'misc', 'van', 'truck', 'tram']
count_labels = np.zeros((len(all_labels),))

g_class2label = {cls: i for i,cls in enumerate(all_labels)}
g_class2color = {'background': [255, 0, 0],
                 'pedestrian':	[0,255,0],
                 'dontcare':	[0,0,255],
                 'car':	[0,255,255],
                 'cyclist':        [255,255,0],
                 'misc':      [255,0,255],
                 'van':      [100,100,255],
                 'truck':        [200,200,100],
                 'tram':       [170,120,200]}
g_easy_view_labels = [7,8,9,10,11,1]
g_label2color = {all_labels.index(cls): g_class2color[cls] for cls in all_labels}

def read_KITTI_label(filename):
    list_of_lists = []
    labels = []
    with open(filename) as f:
        for line in f:
            inner_list = [elt.strip() for elt in line.split(' ')]

            # print(inner_list[0])

            # label = inner_list[0]

            class_label = []
            if inner_list[0] == 'Pedestrian':
                class_label = 0
            elif inner_list[0] == 'Car':
                class_label = 1
            elif inner_list[0] == 'Cyclist':
                class_label = 2
            elif inner_list[0] == 'Van':
                class_label = 3
            elif inner_list[0] == 'Truck':
                class_label = 4
            elif inner_list[0] == 'Tram':
                class_label = 5
            else:
                continue

            count_labels[class_label] += 1
            # print(float(inner_list[5]))
            # list_of_lists.append((float(inner_list[4]), float(inner_list[5]), float(inner_list[6]), float(inner_list[7]), class_label ))
            list_of_lists.append((float(inner_list[8]), float(inner_list[9]), float(inner_list[10]),
                                  float(inner_list[11]), float(inner_list[12]), float(inner_list[13])))
            labels.append(inner_list[0].lower())

    return list_of_lists, labels


def load_velodyne_points(filename):
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    #points = points[:, :3]  # exclude luminance
    return points


def load_calib(calib_dir):
    # P2 * R0_rect * Tr_velo_to_cam * y
    lines = open(calib_dir).readlines()
    lines = [ line.split()[1:] for line in lines ][:-1]
    #
    P = np.array(lines[CAM]).reshape(3,4)
    #
    Tr_velo_to_cam = np.array(lines[5]).reshape(3,4)
    Tr_velo_to_cam = np.concatenate(  [ Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)  ]  , 0     )
    #
    R_cam_to_rect = np.eye(4)
    R_cam_to_rect[:3,:3] = np.array(lines[4][:9]).reshape(3,3)
    #
    P = P.astype('float32')
    Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
    R_cam_to_rect = R_cam_to_rect.astype('float32')
    return P, Tr_velo_to_cam, R_cam_to_rect


def prepare_velo_points(pts3d_raw):
    '''Replaces the reflectance value by 1, and tranposes the array, so
        points can be directly multiplied by the camera projection matrix'''
    pts3d = pts3d_raw
    # Reflectance > 0
    indices = pts3d[:, 3] > 0
    pts3d = pts3d[indices, :]
    # pts3d[:,3] = 1
    return pts3d.transpose(), indices


def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
    '''Project 3D points into 2D image. Expects pts3d as a 4xN
        numpy array. Returns the 2D projection of the points that
        are in front of the camera only an the corresponding 3D points.'''
    # 3D points in camera reference frame.
    pts3d_t = T_cam_velo.dot(pts3d)
    pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))
    # Before projecting, keep only points with z>0
    # (points that are in fronto of the camera).
    idx = (pts3d_cam[2,:]>=0)
    pts2d_cam = Prect.dot(pts3d_cam[:,idx])
    return pts3d_cam[:, idx], pts2d_cam/pts2d_cam[2,:], idx


def get_pcd_from_lidar(pc_dir, calib_dir):

    pts = load_velodyne_points(pc_dir)
    # pts = pcl.load_XYZI(pc_dir)
    P, Tr_velo_to_cam, R_cam_to_rect = load_calib(calib_dir)

    pts3d, indices = prepare_velo_points(pts)
    pts3d_cam, pts2d_normed, idx = project_velo_points_in_img(pts3d, Tr_velo_to_cam, R_cam_to_rect, P)

    # new_x = img_size[1] - pts2d_normed[0, :]
    # new_y = img_size[0] - pts2d_normed[1, :]
    # min_x = min(new_x)
    # max_x = max(new_x)
    # min_y = min(new_y)
    # max_y = max(new_y)
    # img = np.zeros(img_size)
    # for pNo in range(0, pts2d_normed.shape[1]):
    #     x_ind = img_size[1]- int(pts2d_normed[0, pNo])
    #     y_ind = img_size[0] - int(pts2d_normed[1, pNo])
    #     if (x_ind >= 0) & (y_ind >=0) & (x_ind < img_size[1]) & (y_ind < img_size[0]):
    #         img[y_ind, x_ind] = 255

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # pc3d = pcl.PointCloud_PointXYZI()
    pts3d_cam_2 = pts3d_cam.transpose()
    # for pNo in range(0, pts3d_cam.shape[1]):
    #     pts3d_cam_2[0][pNo] = pts3d_cam[0][pNo]# - pts3d_cam[0][0]
    #     pts3d_cam_2[1][pNo] = pts3d_cam[1][pNo]# - pts3d_cam[1][0]
    #     pts3d_cam_2[2][pNo] = pts3d_cam[2][pNo]# - pts3d_cam[2][0]
    # pc3d.from_array(pts3d_cam_2.transpose())

    # path_split = pc_dir.split("/")
    # path_split[7] = 'velodyne_pcd'
    # lidar_file = path_split[8]
    # file = lidar_file.replace('bin', 'pcd')
    # path_split[8] = file
    # pcd_path = "/".join(path_split)
    # pcl.save(pc3d, new_lidar_path)

    # pts3d_ori = pts3d.co
    # py()
    # reflectances = pts[indices, 3]
    # pts3d, pts2d_normed, idx = project_velo_points_in_img(pts3d, Tr_velo_to_cam, R_cam_to_rect, P)
    # # print reflectances.shape, idx.shape
    # reflectances = reflectances[idx]
    # # print reflectances.shape, pts3d.shape, pts2d_normed.shape
    # assert reflectances.shape[0] == pts3d.shape[1] == pts2d_normed.shape[1]

    return pts3d_cam_2


def filter_pts3d(pts3d, limits_x, limits_y, limits_z):
    pts_filtered = []
    # print((min(pts3d[0, :]), max(pts3d[0, :])))
    # print((min(pts3d[1, :]), max(pts3d[1, :])))
    # print((min(pts3d[2, :]), max(pts3d[2, :])))
    for pNo in range(0, pts3d.shape[1]):
        x = pts3d[0][pNo]
        y = pts3d[1][pNo]
        z = pts3d[2][pNo]
        i = pts3d[3][pNo]
        if (x>limits_x[0]) & (x<limits_x[1]) & (y>limits_y[0]) & (y<limits_y[1]) & (z>limits_z[0]) & (z<limits_z[1]):
            pts_filtered.append((x, y, z, i))


    pts3d_filtered = np.array(pts_filtered)
    return  pts3d_filtered


