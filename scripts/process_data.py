#!/usr/bin/env python3
"""Write LiDAR, stereo, and Depth Anything V2 depth to images/ and original main RGB and FARO ground truth to annotations/."""
import argparse
from functools import lru_cache
import json
import shutil
import sys
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from rerun.experimental import RrdReader
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm

from download_data import calibration_files

DATA = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = DATA.parent / "data_processing_models"
STEREO_MATCHERS = ("raft", "sgbm")
RIG = "/world/rig_00"


def transform(q, t):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(q).as_matrix()
    matrix[:3, 3] = t
    return matrix


@lru_cache(maxsize=2)
def load_recording(calibration, info, video):
    """Read the v1 RRD layers once per recording, including the asynchronous rig trajectory."""
    properties = {}
    cameras = [{}, {}]
    trajectories = []
    for path in (info, calibration):
        reader = RrdReader(path)
        if reader.recordings()[0].recording_id != video:
            raise ValueError(f"Calibration recording ID mismatch in {path}")
        for chunk in reader.stream():
            entity = str(chunk.entity_path)
            batch = chunk.to_record_batch()
            if chunk.is_static:
                row = batch.to_pylist()[0]
                if entity.startswith("/__properties/"):
                    properties[entity.split("/")[-1]] = row["value"][0]
                for idx, cam in enumerate(cameras):
                    base = f"{RIG}/cam_{idx:02}"
                    if entity in (base, base + "/pinhole"):
                        cam.update(row)
            elif entity == RIG and "Transform3D:translation" in batch.schema.names:
                ns = batch["video_time"].cast("duration[ns]").cast("int64").to_numpy()
                q = np.asarray(batch["Transform3D:quaternion"].to_pylist())[:, 0]
                t = np.asarray(batch["Transform3D:translation"].to_pylist())[:, 0]
                trajectories.append((ns * 1e-9, q, t))

    if properties["schema_version"] != "arkitscenes:v1" or cameras[1].get("extrinsics_source") != ["mebx"]:
        raise ValueError(f"Unsupported or unmeasured stereo calibration for {video}")
    turns = int(properties["orientation_quarter_turns_ccw"])
    baked_from_native = np.eye(4)
    baked_from_native[:3, :3] = Rotation.from_euler("z", -90 * turns, degrees=True).as_matrix()
    for cam in cameras:
        cam["upright_turns"] = turns
        cam["rig_from_native"] = transform(cam["Transform3D:quaternion"][0],
                                           cam["Transform3D:translation"][0]) @ baked_from_native
        if cam["distortion_source"] != ["mebx_stream_10"]:
            raise ValueError("Expected the v1 Apple stream-10 radial calibration")
        # In this v1 export the eight stream-10 values are the original Apple
        # radial magnification polynomial, despite the generic component label.
        center = np.array(cam["distortion_center_xy"])
        dims = np.array(cam["reference_dimensions_wh"])
        for _ in range(turns % 4):  # undo np.rot90 on reference coordinates
            center = np.array([dims[1] - 1 - center[1], center[0]])
            dims = dims[::-1]
        cam["center"], cam["dims"] = center, dims
        r = np.linspace(0, 1.1, 8192)
        coeff = cam["simplecv.components.DistortionCoefficients"][0]
        scale = 1 + .01 * r**2 * np.polynomial.polynomial.polyval(r**2, coeff)
        if np.any(np.diff(r * scale) <= 0):
            raise ValueError("Lens calibration folds inside the image domain")
        cam["radius"], cam["inverse_scale"] = r * scale, 1 / scale

    times, qs, ts = (np.concatenate(parts) for parts in zip(*trajectories))
    times, idx = np.unique(times, return_index=True)
    # video_time is relative to the recording's uptime epoch.
    times += float(properties["uptime_epoch_seconds"])
    return cameras, times, Slerp(times, Rotation.from_quat(qs[idx])), ts[idx]


def intrinsics(values, size):
    w, h, fx, fy, cx, cy = values
    sx, sy = size[0] / w, size[1] / h
    return np.array([[fx*sx, 0, (cx+.5)*sx-.5], [0, fy*sy, (cy+.5)*sy-.5], [0, 0, 1]])


def maps(size, source_k, rotation, output_k, lens=None, distortion=None):
    x, y = cv2.initUndistortRectifyMap(source_k, distortion, rotation, output_k, size, cv2.CV_32FC1)
    if lens is not None:
        cam, native_size = lens
        ratio = np.array(native_size) / cam["dims"]
        if not np.isclose(*ratio):
            raise ValueError("Lens reference and native image aspect ratios differ")
        center = cam["center"] * ratio
        radius = np.linalg.norm(np.maximum(center, np.array(native_size) - center))
        dx, dy = x - center[0], y - center[1]
        scale = np.interp(np.hypot(dx, dy) / radius, cam["radius"], cam["inverse_scale"])
        x = (center[0] + dx * scale).astype("float32")
        y = (center[1] + dy * scale).astype("float32")
    return x, y


def read(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    return image


def save(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot write {path}")


def refine_pair(main, ultra, depth, meta, cams, prior):
    """Fit small pose and radial corrections using device depth and independent RGB matches."""
    size = (640, 480)
    K = intrinsics(meta["main_lowres_intrinsics"], size)
    sift = cv2.SIFT_create()
    features = []
    for image, values, cam in zip((main, ultra),
            (meta["main_lowres_intrinsics"], meta["ultrawide_intrinsics"]), cams):
        native = image.shape[1::-1]
        mx, my = maps(size, intrinsics(values, native), np.eye(3), K, (cam, native))
        view = cv2.remap(image, mx, my, cv2.INTER_LINEAR)
        features.append(sift.detectAndCompute(cv2.cvtColor(view, cv2.COLOR_BGR2GRAY), None))
    (key1, desc1), (key2, desc2) = features
    summary = {"source": "recorded_rig_and_trajectory"}
    if desc1 is None or desc2 is None or min(len(desc1), len(desc2)) < 2:
        return prior, np.zeros(5), summary
    def matches(a, b):
        return {m.queryIdx: m.trainIdx for m, n in cv2.BFMatcher().knnMatch(a,b,k=2)
                if m.distance < .80*n.distance}
    forward, reverse = matches(desc1, desc2), matches(desc2, desc1)
    pts, observed = [], []
    for i, j in forward.items():
        if reverse.get(j) != i:
            continue
        u,v = key1[i].pt
        z = depth[int((v+.5)*depth.shape[0]/size[1]), int((u+.5)*depth.shape[1]/size[0])] * .001
        if z > 0:
            pts.append([(u-K[0,2])*z/K[0,0], (v-K[1,2])*z/K[1,1], z])
            observed.append(key2[j].pt)
    if len(pts) < 30:
        return prior, np.zeros(5), summary
    pts, observed = np.array(pts), np.array(observed)
    initial = np.linalg.inv(prior)
    start = np.r_[Rotation.from_matrix(initial[:3,:3]).as_rotvec(), 0., 0.]
    fit = np.arange(len(pts)) % 4 != 0
    def error(params):
        # Project into the ultrawide frame with its measured optical center
        # fixed in main-camera coordinates. Only rotation and lens terms vary.
        rotation = Rotation.from_rotvec(params[:3])
        translation = -rotation.apply(prior[:3, 3])
        pred, _ = cv2.projectPoints(pts, params[:3], translation, K,
                                    np.r_[params[3:], 0., 0., 0.])
        return pred[:,0,:] - observed
    def objective(params):
        return np.r_[error(params)[fit].ravel(), params[3:]*2]
    limits = np.array([.1, .1, .1, 1., 1.])
    params = least_squares(objective, start, loss="soft_l1", f_scale=.6,
                           bounds=(start-limits, start+limits)).x
    def heldout(p):
        return float(np.median(np.linalg.norm(error(p)[~fit], axis=1)))
    before, after = heldout(start), heldout(params)
    summary.update(heldout_matches=int(sum(~fit)), evaluation_width_px=size[0],
                   initial_median_reprojection_px=before, fitted_median_reprojection_px=after)
    if after >= before or after > 2.0:
        return prior, np.zeros(5), summary
    matrix = prior.copy()
    matrix[:3, :3] = Rotation.from_rotvec(params[:3]).inv().as_matrix()
    summary["source"] = "recorded_translation_refined_rotation_and_lens"
    return matrix, np.r_[params[3:], 0., 0., 0.], summary



@lru_cache(maxsize=1)
def raft_model():
    """Load Princeton RAFT-Stereo with its SceneFlow stereo checkpoint."""
    import torch
    sys.path.insert(0, str(MODEL_DIR / "RAFT-Stereo"))
    from core.raft_stereo import RAFTStereo
    args = SimpleNamespace(hidden_dims=[128]*3, context_norm="batch", n_downsample=2,
                           n_gru_layers=3, shared_backbone=False, corr_implementation="reg",
                           corr_radius=4, corr_levels=4, slow_fast_gru=False, mixed_precision=False)
    model = RAFTStereo(args)
    state = torch.load(MODEL_DIR / "raftstereo-sceneflow.pth", map_location="cpu", weights_only=True)
    model.load_state_dict({name.removeprefix("module."): value for name, value in state.items()})
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return model.eval().to(device), device


def raft_disparity(left, right):
    import torch
    model, device = raft_model()
    def tensor(image):
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        return torch.from_numpy(rgb).permute(2,0,1)[None].to(device, dtype=torch.float32)
    with torch.inference_mode():
        _, flow = model(tensor(left), tensor(right), iters=24, test_mode=True)
    # Upstream predicts x_right - x_left.
    return -flow[0,0].cpu().numpy()


def refine_photometric(main, ultra, lidar, meta, cams, pose, distortion):
    """Refine rotation, radial distortion and focal scale with the camera center fixed."""
    size = (640, 480)
    k = intrinsics(meta["main_lowres_intrinsics"], size)
    ku = intrinsics(meta["ultrawide_intrinsics"], ultra.shape[1::-1])
    def ideal_gray(image, cam, values, output_size, output_k):
        native = image.shape[1::-1]
        mx, my = maps(output_size, intrinsics(values, native), np.eye(3), output_k, (cam, native))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.remap(gray, mx, my, cv2.INTER_LINEAR).astype(np.float32)

    left = ideal_gray(main, cams[0], meta["main_lowres_intrinsics"], size, k)
    right = ideal_gray(ultra, cams[1], meta["ultrawide_intrinsics"], ultra.shape[1::-1], ku)
    z = cv2.resize(lidar, size, interpolation=cv2.INTER_NEAREST_EXACT).astype(float) * .001
    # Exclude depth discontinuities and give each region of the image a vote.
    spread = cv2.dilate(z, np.ones((7,7),np.uint8)) - cv2.erode(z, np.ones((7,7),np.uint8))
    y,x = np.mgrid[8:size[1]-8:6,8:size[0]-8:6]
    use = (z[y,x] > .25) & (spread[y,x] < .06)
    y,x = y[use],x[use]
    if len(x) < 100:
        return pose, distortion, 1.0
    pts = np.column_stack(((x-k[0,2])/k[0,0], (y-k[1,2])/k[1,1], np.ones_like(x)))
    pts = pts*z[y,x,None] - pose[:3,3]
    initial = np.r_[Rotation.from_matrix(pose[:3,:3].T).as_rotvec(), distortion[:2], 1.]
    params = initial.copy()
    train = ((x//48 + 3*(y//48)) % 5) != 0
    if min(train.sum(), (~train).sum()) < 30:
        return pose, distortion, 1.0

    def normalized(image, blur, window):
        smooth = cv2.GaussianBlur(image, (0,0), blur)
        mean = cv2.GaussianBlur(smooth, (0,0), window)
        var = cv2.GaussianBlur(smooth*smooth, (0,0), window) - mean*mean
        return (smooth-mean) / np.sqrt(np.maximum(var, 16))

    for blur in (2.5, .9):
        target = normalized(left, blur*k[0,0]/ku[0,0], 16)[y,x]
        field = normalized(right, blur, 16*ku[0,0]/k[0,0])
        def residual_and_jacobian(v):
            fitted_k = ku.copy()
            fitted_k[0,0] *= v[5]
            fitted_k[1,1] *= v[5]
            uv, jac = cv2.projectPoints(pts,v[:3],np.zeros(3),fitted_k,np.r_[v[3:5],0.,0.,0.])
            u,w = uv[:,0,:].T
            u = np.clip(u, 0, field.shape[1]-1.001)
            w = np.clip(w, 0, field.shape[0]-1.001)
            i,j = u.astype(int),w.astype(int)
            a,b = u-i,w-j
            top = field[j,i]*(1-a) + field[j,i+1]*a
            bottom = field[j+1,i]*(1-a) + field[j+1,i+1]*a
            dx = (1-b)*(field[j,i+1]-field[j,i]) + b*(field[j+1,i+1]-field[j+1,i])
            jac = jac.reshape(len(u),2,-1)
            focal_deriv = jac[:,:,6]*ku[0,0] + jac[:,:,7]*ku[1,1]
            deriv = np.dstack((jac[:,:,[0,1,2,10,11]], focal_deriv))
            return (1-b)*top+b*bottom-target, dx[:,None]*deriv[:,0]+(bottom-top)[:,None]*deriv[:,1]

        span = np.array([.05,.05,.05,1.,1.,.12])
        fit = least_squares(lambda v: residual_and_jacobian(v)[0][train], params,
                            jac=lambda v: residual_and_jacobian(v)[1][train],
                            bounds=(initial-span,initial+span), loss="soft_l1", f_scale=.3, max_nfev=25)
        params = fit.x
    before = np.mean(np.minimum(abs(residual_and_jacobian(initial)[0][~train]), 1))
    after = np.mean(np.minimum(abs(residual_and_jacobian(params)[0][~train]), 1))
    if after >= before:
        return pose, distortion, 1.0
    result = pose.copy()
    result[:3,:3] = Rotation.from_rotvec(params[:3]).inv().as_matrix()
    return result, np.r_[params[3:5],0.,0.,0.], params[5]


def sgbm_disparity(views, masks, focal_baseline):
    """Match both directions and verify native disparities against a half-scale SGBM estimate."""
    def gray(image):
        # Match the two cameras at a more similar optical bandwidth while
        # retaining fractional shifts on the full-resolution pixel grid.
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    left = cv2.GaussianBlur(gray(views[0]), (0, 0), 1.8)
    right = gray(views[1])
    def normalize(gray):
        pixels = gray.astype(np.float32)
        mean = cv2.GaussianBlur(pixels, (0, 0), 8)
        variance = cv2.GaussianBlur(pixels*pixels, (0, 0), 8) - mean*mean
        return np.clip(128 + 32*(pixels-mean)/np.sqrt(np.maximum(variance, 25)), 0, 255).astype(np.uint8)

    a, b = normalize(left), normalize(right)
    count = max(16, 16 * int(np.ceil((focal_baseline / .25 + 2) / 16)))
    count = min(count, (left.shape[1] // 32) * 16)
    block = 45
    engine = cv2.StereoSGBM_create(minDisparity=0, numDisparities=count, blockSize=block,
                                   P1=block**2, P2=4*block**2, preFilterCap=15,
                                   uniquenessRatio=8, speckleWindowSize=100, speckleRange=1,
                                   disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    # Scale the block-summed matching costs for OpenCV's 16-bit SGBM paths.
    match_a, match_b = a // 8, b // 8
    d = engine.compute(match_a, match_b).astype(np.float32) / 16
    rev = engine.compute(match_b[:, ::-1], match_a[:, ::-1])[:, ::-1].astype(np.float32) / 16
    yy, xx = np.indices(d.shape, dtype=np.float32)
    match_x = xx - d
    opposite = cv2.remap(rev, match_x, yy, cv2.INTER_LINEAR)
    supported = cv2.remap(masks[1].astype(np.float32), match_x, yy, cv2.INTER_LINEAR)
    # Repeat matching at half scale (larger physical support). Independent
    # agreement helps reject repeated patterns and poorly localized edges.
    small_a, small_b = (cv2.pyrDown(image) for image in (match_a, match_b))
    engine.setNumDisparities(16 * int(np.ceil(count / 32)))
    coarse = engine.compute(small_a, small_b).astype(np.float32) / 8
    coarse = cv2.resize(coarse, (d.shape[1], d.shape[0]), interpolation=cv2.INTER_LINEAR)
    support = masks[0] & (supported > .999)
    valid = support & (d > 0) & (coarse > 0) & (opposite > 0)
    valid &= (np.abs(d-opposite) <= .75) & (np.abs(d-coarse) <= 2.)

    # Average normalized appearance error only over pixels with correspondences.
    # Invalid disparities in a textured patch must not displace its other pixels.
    window = (25, 25)
    warped = cv2.remap(b, match_x, yy, cv2.INTER_LINEAR)
    residual = cv2.absdiff(a, warped).astype(np.float32)
    weight = (support & (d>0)).astype(np.float32)
    fraction = cv2.blur(weight, window)
    photo_error = cv2.blur(weight*residual, window) / np.maximum(fraction, 1e-6)
    # A physical uncertainty gate: a measurable full-resolution
    # disparity change should separate this estimate from one 25 cm farther.
    delta = .25*d**2 / (float(focal_baseline) + .25*np.maximum(d, 0))
    grad = np.abs(cv2.Sobel(left, cv2.CV_32F, 1, 0, ksize=3, scale=1/8))
    valid &= (fraction > .20) & (photo_error < 27) & (cv2.blur(grad, window) > .45)
    valid &= delta > .25
    return np.where(valid, d, 0)


def calibrate_disparity(disparity, lidar, main_intrinsics, rect, k, focal_baseline):
    """Estimate an image-wide affine horizontal calibration error in rectified pixels.

    Each fit uses at most one LiDAR ray per 2x2 sensor cells. Spatially held-out
    cells measure residual uncertainty. No pixel receives a LiDAR depth value.
    """
    h, w = disparity.shape
    kd = intrinsics(main_intrinsics, lidar.shape[1::-1])
    v, u = np.mgrid[1:lidar.shape[0]:2, 1:lidar.shape[1]:2]
    rays = np.stack(((u-kd[0,2])/kd[0,0], (v-kd[1,2])/kd[1,1], np.ones_like(u)))
    rotated = (rect @ rays.reshape(3,-1)).reshape(rays.shape)
    factor = rotated[2]
    x = np.rint(k[0,0]*rotated[0]/factor+k[0,2]).astype(int)
    y = np.rint(k[1,1]*rotated[1]/factor+k[1,2]).astype(int)
    good = (factor>0) & (lidar[v,u]>0) & (x>=0) & (x<w) & (y>=0) & (y<h)
    x,y,u,v,factor = (item[good] for item in (x,y,u,v,factor))
    measured = disparity[y,x]
    good = measured > 0
    x,y,u,v,factor,measured = (item[good] for item in (x,y,u,v,factor,measured))
    design = np.column_stack((np.ones_like(x), 2*x/w-1, 2*y/h-1))
    error = measured - 1000*focal_baseline/(factor*lidar[v,u])
    train = ((u//24 + 3*(v//24)) % 5) != 0
    if train.sum() < 24 or (~train).sum() < 12:
        # No independent evidence to certify metric precision for this pair.
        return np.zeros_like(disparity), np.inf
    coeff = np.zeros(3)
    for _ in range(6):
        weights = train * np.minimum(1., 1./np.maximum(np.abs(error-design@coeff), 1e-6))
        coeff = np.linalg.solve(design.T@(weights[:,None]*design) + np.eye(3)*2,
                                design.T@(weights*error))
    heldout = error[~train]
    if np.median(abs(heldout-design[~train]@coeff)) >= np.median(abs(heldout)):
        coeff[:] = 0
    uncertainty = max(.5, np.percentile(np.abs(heldout-design[~train]@coeff), 80))
    yy,xx = np.ogrid[:h,:w]
    correction = coeff[0] + coeff[1]*(2*xx/w-1) + coeff[2]*(2*yy/h-1)
    corrected = np.where(disparity>0, np.maximum(disparity-correction,0), 0).astype(np.float32)
    return corrected, uncertainty


def rectified_depth(main, ultra, lidar, meta, cams, main_from_ultra, distortion, matcher):
    """Rectify working copies and return z-depth on the original main-view grid."""
    if matcher not in STEREO_MATCHERS:
        raise ValueError(f"Unknown stereo matcher {matcher!r}. Choose from {STEREO_MATCHERS}.")
    size = main.shape[1::-1]
    output_size = size
    k = intrinsics(meta["main_lowres_intrinsics"], size)
    baseline = main_from_ultra[:3,3]
    length = np.linalg.norm(baseline)
    if length < 1e-6:
        raise ValueError("Cannot triangulate with a zero baseline")
    # Keep the native view upright, then mirror both network inputs if the
    # ultrawide optical center lies left of the main camera.
    x = baseline / length
    mirror = x[0] < 0
    if mirror:
        x = -x
    y = np.cross(np.array([0.,0.,1.]) + main_from_ultra[:3,2], x)
    y /= np.linalg.norm(y)
    rect = np.stack((x, y, np.cross(x,y)))
    # When the device moves forward between exposures, the rectified optical
    # axis can point well outside both images. Center the virtual image on the
    # main viewing ray and compensate for its local projective magnification.
    ray_x, ray_y, ray_z = rect[:,2]
    if ray_z <= 1e-5:
        return np.zeros(output_size[::-1], dtype=np.uint16)
    k[0,0] *= ray_z**2
    k[1,1] *= ray_z
    k[0,2] -= k[0,0]*ray_x/ray_z
    k[1,2] -= k[1,1]*ray_y/ray_z
    views, masks = [], []
    for image, values, cam, rotation, lens in zip(
        (main, ultra), (meta["main_lowres_intrinsics"], meta["ultrawide_intrinsics"]), cams,
        (rect, rect @ main_from_ultra[:3,:3]), (None, distortion)
    ):
        native = image.shape[1::-1]
        mx,my = maps(size, intrinsics(values,native), rotation, k, (cam,native), lens)
        image = cv2.remap(image,mx,my,cv2.INTER_LINEAR)
        mask = (mx>=0)&(mx<native[0]-1)&(my>=0)&(my<native[1]-1)
        views.append(image[:,::-1] if mirror else image)
        masks.append(mask[:,::-1] if mirror else mask)
    focal_baseline = k[0,0]*length
    if matcher == "sgbm":
        disparity = sgbm_disparity(views, masks, focal_baseline)
    else:
        # A known principal-point offset moves this short-baseline pair into a
        # larger disparity range for the network. Undo it before triangulation.
        offset = 32
        views[1] = cv2.copyMakeBorder(views[1][:,offset:],0,0,0,offset,cv2.BORDER_REPLICATE)
        masks[1] = np.pad(masks[1][:,offset:], ((0,0),(0,offset)))
        disparity = raft_disparity(*views)
        reverse = raft_disparity(views[1][:,::-1], views[0][:,::-1])[:,::-1]
        yy,xx = np.indices(disparity.shape,dtype=np.float32)
        match_x = xx-disparity
        other = cv2.remap(reverse,match_x,yy,cv2.INTER_LINEAR)
        supported = cv2.remap(masks[1].astype(np.float32),match_x,yy,cv2.INTER_LINEAR) > .999
        valid = masks[0] & supported & (disparity>0) & (other>0) & (abs(disparity-other)<1)
        disparity = disparity - offset
        disparity = np.where(valid, disparity, 0)
    if mirror:
        disparity = disparity[:,::-1]
    if matcher == "sgbm":
        disparity, uncertainty = calibrate_disparity(
            disparity, lidar, meta["main_lowres_intrinsics"], rect, k, focal_baseline)
    inverse_z = (disparity / focal_baseline).astype(np.float32)
    # A unit-z ray on the output grid rotates to a rectified ray of
    # axial length factor. Thus native z = rectified z / factor.
    kd = intrinsics(meta["main_lowres_intrinsics"],output_size)
    v,u = np.indices(output_size[::-1], dtype=np.float32)
    rays = np.stack(((u-kd[0,2])/kd[0,0], (v-kd[1,2])/kd[1,1], np.ones_like(u)))
    directions = (rect @ rays.reshape(3,-1)).reshape(rays.shape)
    factor = directions[2]
    mx = (k[0,0]*directions[0]/factor+k[0,2]).astype(np.float32)
    my = (k[1,1]*directions[1]/factor+k[1,2]).astype(np.float32)
    inv_native = cv2.remap(inverse_z,mx,my,cv2.INTER_NEAREST)*factor
    valid = (inv_native>=.1)&(inv_native<=4)
    if matcher == "sgbm":
        # Propagate the held-out disparity residual into native axial depth.
        # For a candidate z, require even the farther bound to lie within 25 cm.
        max_inv_error = .25 * inv_native**2 / (1 + .25*np.maximum(inv_native, 0))
        valid &= (factor > 0) & (max_inv_error >= uncertainty * factor / focal_baseline)
    depth = np.zeros(v.shape,np.uint16)
    depth[valid] = np.rint(1000/inv_native[valid]).astype(np.uint16)
    return depth


def save_depth_preview(path, depth, low, high):
    """Colorize millimeter depths on the sample's shared scale."""
    intensity = np.clip((depth.astype(np.float32) - low) * (255 / (high - low)), 0, 255)
    preview = cv2.applyColorMap(intensity.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    preview[depth == 0] = 0
    save(path, preview)


@lru_cache(maxsize=1)
def depthanything_model():
    """Load the official ViT-S indoor metric checkpoint (Hypersim, 20 m range)."""
    import torch
    sys.path.insert(0, str(MODEL_DIR / "Depth-Anything-V2" / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384],
                            max_depth=20)
    state = torch.load(MODEL_DIR / "depth_anything_v2_metric_hypersim_vits.pth",
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return model.to(device).eval()


def save_depthanything(main, turns, folder, low, high):
    upright = np.ascontiguousarray(np.rot90(main, turns))
    meters = np.rot90(depthanything_model().infer_image(upright), -turns)
    if meters.shape != main.shape[:2] or not np.isfinite(meters).all() or meters.max() <= 0:
        raise ValueError("Depth Anything returned invalid metric depth")
    depth = np.rint(np.clip(meters, 0, 20) * 1000).astype(np.uint16)
    with TemporaryDirectory(dir=folder) as temporary:
        stage = Path(temporary)
        png = stage / "depthanything_depth.png"
        preview = stage / "depthanything_depth_preview.jpeg"
        save(png, depth)
        save_depth_preview(preview, depth, low, high)
        png.replace(folder / png.name)
        preview.replace(folder / preview.name)


def estimate_sample(source, matcher):
    meta = json.loads((source / "sample.json").read_text())
    files = calibration_files(source.parent, meta["video_id"], source)
    cams, times, rotations, translations = load_recording(*files, meta["video_id"])
    world_from_cam = []
    for timestamp, cam in zip((meta["timestamp_s"], meta["ultrawide_timestamp_s"]), cams):
        if timestamp < times[0] - .02 or timestamp > times[-1] + .02:
            raise ValueError(f"{source.name}: camera exposure outside rig trajectory")
        t = np.clip(timestamp, times[0], times[-1])
        position = [np.interp(t, times, col) for col in translations.T]
        world_from_cam.append(transform(rotations(t).as_quat(), position) @ cam["rig_from_native"])
    # This transform maps ultrawide rays into the main frame at its exposure.
    main_from_ultra = np.linalg.solve(world_from_cam[0], world_from_cam[1])
    main, ultra = (read(source / name) for name in ("main.png", "ultrawide.png"))
    lidar = read(source / "lidar.png")
    main_from_ultra, distortion, quality = refine_pair(main, ultra, lidar, meta, cams, main_from_ultra)
    main_from_ultra, distortion, focal_scale = refine_photometric(
        main, ultra, lidar, meta, cams, main_from_ultra, distortion)
    for i in (2, 3):
        meta["ultrawide_intrinsics"][i] *= focal_scale
    depth = rectified_depth(main, ultra, lidar, meta, cams, main_from_ultra, distortion, matcher)
    return depth, main, lidar, cams[0]["upright_turns"], quality


def process_sample(source, output, matcher="raft", stereo_only=False):
    depth, main, lidar, turns, _ = estimate_sample(source, matcher)
    images = output / "images" / source.name
    annotations = output / "annotations" / source.name
    with TemporaryDirectory(dir=output) as temporary:
        stage = Path(temporary)
        valid = lidar[lidar > 0]
        low, high = np.percentile(valid, [2, 98]) if valid.size else (0, 10000)
        high = max(high, low + 1)
        images.mkdir(parents=True, exist_ok=True)
        save(stage / "stereo_depth.png", depth)
        save_depth_preview(stage / "stereo_depth_preview.jpeg", depth, low, high)
        if stereo_only:
            for file in stage.iterdir():
                file.replace(images / file.name)
            return
        shutil.copyfile(source / "lidar.png", stage / "lidar.png")
        shutil.copyfile(source / "ground_truth.png", stage / "ground_truth.png")
        annotations.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "main.png", stage / "main.png")
        (stage / "main.png").replace(annotations / "main.png")
        for name in ("stereo_depth.png", "stereo_depth_preview.jpeg"):
            (stage / name).replace(images / name)
        save_depthanything(main, turns, images, low, high)
        for stem, pixels, dest in (("lidar", lidar, images),
                                   ("ground_truth", read(stage / "ground_truth.png"), annotations)):
            preview = stage / f"{stem}_preview.jpeg"
            save_depth_preview(preview, pixels, low, high)
            png = stage / f"{stem}.png"
            png.replace(dest / png.name)
            preview.replace(dest / preview.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA / "raw")
    parser.add_argument("--output", type=Path, default=DATA / "processed_data")
    parser.add_argument("--matcher", choices=STEREO_MATCHERS, default="raft")
    parser.add_argument("--stereo-only", action="store_true",
                        help="Regenerate just stereo_depth.png and its preview for every pair")
    args = parser.parse_args()
    samples = sorted(p.parent for p in args.input.glob("*/sample.json"))
    if not samples:
        parser.exit(1, f"No raw samples in {args.input}\n")
    if args.input.resolve() == args.output.resolve():
        parser.error("Use separate raw and processed directories")
    args.output.mkdir(parents=True, exist_ok=True)
    stereo = f"{args.matcher.upper()} stereo depth"
    if args.stereo_only:
        description = f"Stereo only ({stereo})"
    else:
        description = f"Full samples ({stereo}, Depth Anything V2, LiDAR, annotations)"
    for source in tqdm(samples, desc=description, unit="sample", dynamic_ncols=True):
        process_sample(source, args.output, args.matcher, args.stereo_only)


if __name__ == "__main__":
    main()
