import numpy as np
import open3d as o3d


# noinspection PyPep8Naming
def map_3d_to_2d_project(points3d, K, R, t, w, h, new_w, new_h):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # [R|t] transform XYZ_world to XYZ_cam
    # colmap pose: from world to camera
    pts_cam = np.matmul(R, points3d.transpose()) + t[:, np.newaxis]
    pts_cam = pts_cam.transpose()

    # get the depth value
    # depth_values = pts_cam[:, 2]

    # project the 3d points to 2d pixel coordinate
    # 2D normalized + multiply the intrinsic matrix (K)
    x_norm = pts_cam[:, 0] / pts_cam[:, 2]
    y_norm = pts_cam[:, 1] / pts_cam[:, 2]
    assert len(np.nonzero(pts_cam[:, 2] == 0)) != 0

    new_fx = fx * (new_w / w)
    new_fy = fy * (new_h / h)
    new_cx = cx * (new_w / w)
    new_cy = cy * (new_h / h)
    x_2d = x_norm * new_fx + new_cx
    y_2d = y_norm * new_fy + new_cy

    x_2d = np.round(x_2d).astype(np.int32)
    y_2d = np.round(y_2d).astype(np.int32)
    points2d = np.array([x_2d, y_2d]).transpose()

    invalid_indices = list()
    for i, (x, y) in enumerate(zip(x_2d, y_2d)):
        if (x < 0) or (y < 0) or (x >= new_w) or (y >= new_h):
            invalid_indices.append(i)

    invalid_indices = np.asarray(invalid_indices, dtype=np.int32)

    return points2d, invalid_indices


# noinspection PyPep8Naming
def map_2d_to_3d_project(points2d, K, R, t, depths, scale=1.):
    # points2d : n x 2 array of pixel locations in an image
    # K : Intrinsic matrix for camera
    # R : Rotation matrix describing rotation of camera frame
    #     w.r.t world frame.
    # t : Translation vector describing the translation of camera frame
    #     w.r.t world frame
    # [R t] combined is known as the Camera Pose.

    R = R.T
    t = -R @ t
    t.shape = (3, 1)

    points3d = list()
    for i, (p, d) in enumerate(zip(points2d, depths)):
        # Homogeneous pixel coordinate
        p = np.array([p[0], p[1], 1]).T
        p.shape = (3, 1)

        # Transform pixel in Camera coordinate frame
        pc = np.linalg.inv(K) @ p

        # Transform pixel in World coordinate frame
        pw = t + (R @ pc)

        # Transform camera origin in World coordinate frame
        cam = np.array([0, 0, 0]).T
        cam.shape = (3, 1)
        cam_world = t + R @ cam

        # Find a ray from camera to 3d point
        vector = pw - cam_world
        unit_vector = vector / np.linalg.norm(vector)

        # Point scaled along this ray
        p3d = cam_world + scale * d * unit_vector
        points3d.append(p3d.squeeze(1))

    return np.asarray(points3d)


# noinspection PyPep8Naming
def map_2d_to_3d_colmap(points2d, masks, image, points3D, scale=1.):
    points3d_indices_for_img = image.point3D_ids

    points3d = list()
    feature_pts = list()
    for i, coord in enumerate(image.xys):
        point2d_scale = (coord * scale).astype(np.int32)

        if points3d_indices_for_img[i] > -1 and masks[point2d_scale[1], point2d_scale[0]] == 1:
            points3d.append(points3D[points3d_indices_for_img[i]].xyz)
            feature_pts.append(point2d_scale)

    points3d = np.asarray(points3d)
    feature_pts = np.asarray(feature_pts)

    dists = np.empty((len(points2d), len(points3d)), dtype=np.float64)
    for i, point2d in enumerate(points2d):
        dists[i] = np.linalg.norm((point2d - feature_pts), axis=1)
    sort_indices = np.argsort(dists, axis=1)

    return points3d, sort_indices, feature_pts


def parse_cam(cam):
    camera_param = cam.params

    if cam.model == 'PINHOLE':
        fx = camera_param[0]
        fy = camera_param[1]
        cx = camera_param[2]
        cy = camera_param[3]
        w = cam.width
        h = cam.height

    elif cam.model == 'SIMPLE_RADIAL':
        fx = camera_param[0]
        fy = camera_param[0]
        cx = camera_param[1]
        cy = camera_param[2]
        w = cam.width
        h = cam.height

    elif cam.model == 'SIMPLE_PINHOLE':
        fx = camera_param[0]
        fy = camera_param[0]
        cx = camera_param[1]
        cy = camera_param[2]
        w = cam.width
        h = cam.height

    else:
        raise RuntimeError(f'Undefined cam model {cam.model}')

    assert fx == fy, 'nerf cam format asks focal x equals to y'
    assert cx == w / 2.
    assert cy == h / 2.

    return fx, fy, cx, cy, w, h


# noinspection PyPep8Naming
def gen_cam_param_colmap(cam, img):
    fx, fy, cx, cy, w, h = parse_cam(cam)

    K = np.eye(3)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    R = img.qvec2rotmat()
    t = img.tvec

    return K, R, t, w, h


# noinspection PyPep8Naming
def gen_cam_param_transform(camera_angle_x, transform_matrix, img_shape):
    h, w = img_shape
    focal = 0.5 * w / np.tan(0.5 * float(camera_angle_x))
    K = np.array(
        [
            [focal, 0, 0.5 * w],
            [0, focal, 0.5 * h],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    w2c_mats = transform_matrix
    c2w_mats = np.linalg.inv(w2c_mats)

    R = c2w_mats[:3, :3]
    t = c2w_mats[:3, 3]

    return K, R, t, w, h


def draw_pcd(points, colors, out_path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, pcd)
