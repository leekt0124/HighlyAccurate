import torch
import numpy as np
import cv2

from pathlib import Path
from functools import lru_cache

from pyquaternion import Quaternion
from shapely.geometry import MultiPolygon

from .common import INTERPOLATION, get_view_matrix, get_pose, get_split, get_yaw
from .transforms import Sample, SaveDataTransform

from PIL import Image
import os
import pandas as pd
import utils
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.utils import save_image



import sys
sys.path.append("..")
from googleMapDownloader_nuscenes import GoogleMapDownloader, GoogleMapsLayers, getLatLongfromPose, MAP_ORIGINS
 

STATIC = ['lane', 'road_segment']
DIVIDER = ['road_divider', 'lane_divider']
DYNAMIC = [
    'car', 'truck', 'bus',
    'trailer', 'construction',
    'pedestrian',
    'motorcycle', 'bicycle',
]

CLASSES = STATIC + DIVIDER + DYNAMIC
NUM_CLASSES = len(CLASSES)


def get_data(
    version,  # 'v1.0-trainval' or -mini
    dataset_dir,
    labels_dir,
    input_image_transform,
    split,
    shift_range_lat, shift_range_lon, rotation_range, 
    root_dir,
    zoom_level,
   
    dataset='unused',                   # ignore
    augment='unused',                   # ignore
    image='unused',                     # ignore
    label_indices='unused',             # ignore
    num_classes=NUM_CLASSES,            # in here to make config consistent
    **dataset_kwargs
):
    assert num_classes == NUM_CLASSES

    helper = NuScenesSingleton(dataset_dir, version)
    transform = SaveDataTransform(labels_dir)

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    # print(f'In get_data(): split: {split}')
    split_scenes = get_split(split, 'nuscenes')

    result = list()
    for scene_name, scene_record in helper.get_scenes():
        if scene_name not in split_scenes:
            continue

        data = NuScenesDataset(scene_name, scene_record, zoom_level, helper, input_image_transform,
                               transform, shift_range_lat, shift_range_lon, rotation_range, root_dir, **dataset_kwargs)
        result.append(data)

    return result


def get_split_data(version, dataset_dir, labels_dir, transform, shift_range_lat, shift_range_lon, rotation_range, \
                   root_dir, zoom_level, loader_config, split, loader=True, shuffle=False):
    # get a list of NuScenesDataset
    datasets = get_data(
        version, # v1.0-trainval or v1.0-mini
        dataset_dir,
        labels_dir,
        transform,
        split,
        shift_range_lat,
        shift_range_lon,
        rotation_range,
        root_dir,
        zoom_level
        )

    if not loader:
        return datasets

    # Concatenate a list of NuScenesDataset => one dataset
    dataset = torch.utils.data.ConcatDataset(datasets)
    # print("len(dataset) = ", len(dataset))

    loader_config = dict(loader_config)

    # if loader_config['num_workers'] == 0:
        # loader_config['prefetch_factor'] = 2
    # return a DataLoader as "train" or "val" dataloader 
    return torch.utils.data.DataLoader(dataset, shuffle=shuffle, **loader_config)


def get_satmap_name(filename):

    # input filename:
    # scene-0001/CAM_BACK_LEFT/n015-2018-07-18-11-07-57+0800__CAM_BACK_LEFT__1531883530447423.jpg

    # split by / to get the last item
    last_item = filename.split('/')[-1]
    # Replace image_name by __satmap__
    tmp_list = last_item.split('__')
    tmp_list[1] = "_satmap_"
    # Convert a list to a string and return
    satmap_name = ''.join(tmp_list)
    return satmap_name


class NuScenesSingleton:
    """
    Wraps both nuScenes and nuScenes map API

    This was an attempt to sidestep the 30 second loading time in a "clean" manner
    """
    def __init__(self, dataset_dir, version):
        """
        dataset_dir: /path/to/nuscenes/
        version: v1.0-trainval
        """
        self.dataroot = str(dataset_dir)
        self.nusc = self.lazy_nusc(version, self.dataroot)

    @classmethod
    def lazy_nusc(cls, version, dataroot):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.nuscenes import NuScenes

        if not hasattr(cls, '_lazy_nusc'):
            cls._lazy_nusc = NuScenes(version=version, dataroot=dataroot)

        return cls._lazy_nusc

    def get_scenes(self):
        for scene_record in self.nusc.scene:
            yield scene_record['name'], scene_record

    @lru_cache(maxsize=16)
    def get_map(self, log_token):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.map_expansion.map_api import NuScenesMap

        map_name = self.nusc.get('log', log_token)['location']
        nusc_map = NuScenesMap(dataroot=self.dataroot, map_name=map_name)

        return nusc_map

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, '_singleton'):
            obj = super(NuScenesSingleton, cls).__new__(cls)
            obj.__init__(*args, **kwargs)

            cls._singleton = obj

        return cls._singleton


class NuScenesDataset(torch.utils.data.Dataset):
    CAMERAS = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
               'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']

    def __init__(
        self,
        scene_name: str,
        scene_record: dict,
        zoom_level: int,
        helper: NuScenesSingleton,
        input_image_transform,
        transform=None,                 # transform: the label class (SaveDataTransform)
        shift_range_lat=20, shift_range_lon=20, rotation_range=10,
        root_dir = "/mnt/workspace/datasets/nuscenes/",
        cameras=[[0, 1, 2, 3, 4, 5]],
        bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0}
    ):
        self.scene_name = scene_name
        self.zoom_level = zoom_level # for download satmap

        self.transform = transform

        self.nusc = helper.nusc
        self.nusc_map = helper.get_map(scene_record['log_token'])

        self.view = get_view_matrix(**bev)
        self.bev_shape = (bev['h'], bev['w'])

        self.samples = self.parse_scene(scene_record, cameras)

        # Added by Goro
        if input_image_transform != None:
            self.satmap_transform = input_image_transform[0]
            self.grdimage_transform = input_image_transform[1]

        # self.meter_per_pixel = utils.get_meter_per_pixel(scale=1)
        self.shift_range_meters_lat = shift_range_lat  # in terms of meters
        self.shift_range_meters_lon = shift_range_lon  # in terms of meters
        self.rotation_range = rotation_range  # in terms of degree

        # For __getitem__() to load /samples/ and /satmap/
        self.root_dir = root_dir

    def parse_scene(self, scene_record, camera_rigs):
        data = []
        sample_token = scene_record['first_sample_token']

        while sample_token:
            sample_record = self.nusc.get('sample', sample_token)

            for camera_rig in camera_rigs:
                data.append(self.parse_sample_record(sample_record, camera_rig))

            sample_token = sample_record['next']

        return data

    def parse_pose(self, record, *args, **kwargs):
        return get_pose(record['rotation'], record['translation'], *args, **kwargs)

    def get_yaw_in_radian(self, record):
        return get_yaw(record['rotation'])
    
    def parse_sample_record(self, sample_record, camera_rig):
        lidar_record = self.nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
        egolidar = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])

        world_from_egolidarflat = self.parse_pose(egolidar, flat=True)
        egolidarflat_from_world = self.parse_pose(egolidar, flat=True, inv=True)

        cam_channels = []
        images = []
        intrinsics = []
        extrinsics = []

        for cam_idx in camera_rig:
            cam_channel = self.CAMERAS[cam_idx]
            cam_token = sample_record['data'][cam_channel]

            cam_record = self.nusc.get('sample_data', cam_token)
            egocam = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
            cam = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])

            cam_from_egocam = self.parse_pose(cam, inv=True)
            egocam_from_world = self.parse_pose(egocam, inv=True)

            E = cam_from_egocam @ egocam_from_world @ world_from_egolidarflat
            I = cam['camera_intrinsic']

            full_path = Path(self.nusc.get_sample_data_path(cam_token))
            # print(f'NuSceneDataset parse_sample_record full_path {full_path}')
            image_path = str(full_path.relative_to(self.nusc.dataroot))
            # print(f'image_path {image_path}')

            cam_channels.append(cam_channel)
            intrinsics.append(I)
            extrinsics.append(E.tolist())
            images.append(image_path)

            yaw = self.get_yaw_in_radian(egolidar)

            # Get nusc.pose 
            if cam_channel=="CAM_FRONT":
                pose = self.nusc.get('ego_pose', cam_record['ego_pose_token'])

        # Get log
        scene_record = self.nusc.get('scene', sample_record['scene_token'])
        log = self.nusc.get('log', scene_record['log_token'])
        location = log["location"]

        return {
            'scene': self.scene_name,
            'token': sample_record['token'],

            'location': location,
            'pose_for_satmap'    : pose,

            'pose': world_from_egolidarflat.tolist(),
            'pose_inverse': egolidarflat_from_world.tolist(),

            'yaw' : yaw,

            'cam_ids': list(camera_rig),
            'cam_channels': cam_channels,
            'intrinsics': intrinsics,
            'extrinsics': extrinsics,
            'images': images,
        }

    def get_dynamic_objects(self, sample, annotations):
        h, w = self.bev_shape[:2]

        segmentation = np.zeros((h, w), dtype=np.uint8)
        center_score = np.zeros((h, w), dtype=np.float32)
        center_offset = np.zeros((h, w, 2), dtype=np.float32)
        center_ohw = np.zeros((h, w, 4), dtype=np.float32)
        buf = np.zeros((h, w), dtype=np.uint8)

        visibility = np.full((h, w), 255, dtype=np.uint8)

        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32)

        for ann, p in zip(annotations, self.convert_to_box(sample, annotations)):
            box = p[:2, :4]
            center = p[:2, 4]
            front = p[:2, 5]
            left = p[:2, 6]

            buf.fill(0)
            cv2.fillPoly(buf, [box.round().astype(np.int32).T], 1, INTERPOLATION)
            mask = buf > 0

            if not np.count_nonzero(mask):
                continue

            sigma = 1
            segmentation[mask] = 255
            center_offset[mask] = center[None] - coords[mask]
            center_score[mask] = np.exp(-(center_offset[mask] ** 2).sum(-1) / (sigma ** 2))

            # orientation, h/2, w/2
            center_ohw[mask, 0:2] = ((front - center) / (np.linalg.norm(front - center) + 1e-6))[None]
            center_ohw[mask, 2:3] = np.linalg.norm(front - center)
            center_ohw[mask, 3:4] = np.linalg.norm(left - center)

            visibility[mask] = ann['visibility_token']
 
        segmentation = np.float32(segmentation[..., None])
        center_score = center_score[..., None]

        result = np.concatenate((segmentation, center_score, center_offset, center_ohw), 2)

        # (h, w, 1 + 1 + 2 + 2)
        return result, visibility

    def convert_to_box(self, sample, annotations):
        # Import here so we don't require nuscenes-devkit unless regenerating labels
        from nuscenes.utils import data_classes

        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        for a in annotations:
            box = data_classes.Box(a['translation'], a['size'], Quaternion(a['rotation']))

            corners = box.bottom_corners()                                              # 3 4
            center = corners.mean(-1)                                                   # 3
            front = (corners[:, 0] + corners[:, 1]) / 2.0                               # 3
            left = (corners[:, 0] + corners[:, 3]) / 2.0                                # 3

            p = np.concatenate((corners, np.stack((center, front, left), -1)), -1)      # 3 7
            p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)                        # 4 7
            p = V @ S @ M_inv @ p                                                       # 3 7

            yield p                                                                     # 3 7

    def get_category_index(self, name, categories):
        """
        human.pedestrian.adult
        """
        tokens = name.split('.')

        for i, category in enumerate(categories):
            if category in tokens:
                return i

        return None

    def get_annotations_by_category(self, sample, categories):
        result = [[] for _ in categories]

        for ann_token in self.nusc.get('sample', sample['token'])['anns']:
            a = self.nusc.get('sample_annotation', ann_token)
            idx = self.get_category_index(a['category_name'], categories)

            if idx is not None:
                result[idx].append(a)

        return result

    def get_line_layers(self, sample, layers, patch_radius=150, thickness=1):
        h, w = self.bev_shape[:2]
        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        box_coords = (sample['pose'][0][-1] - patch_radius, sample['pose'][1][-1] - patch_radius,
                      sample['pose'][0][-1] + patch_radius, sample['pose'][1][-1] + patch_radius)
        records_in_patch = self.nusc_map.get_records_in_patch(box_coords, layers, 'intersect')

        result = list()

        for layer in layers:
            render = np.zeros((h, w), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map.get(layer, r)
                line = self.nusc_map.extract_line(polygon_token['line_token'])

                p = np.float32(line.xy)                                     # 2 n
                p = np.pad(p, ((0, 1), (0, 0)), constant_values=0.0)        # 3 n
                p = np.pad(p, ((0, 1), (0, 0)), constant_values=1.0)        # 4 n
                p = V @ S @ M_inv @ p                                       # 3 n
                p = p[:2].round().astype(np.int32).T                        # n 2

                cv2.polylines(render, [p], False, 1, thickness=thickness)

            result.append(render)

        return 255 * np.stack(result, -1)

    def get_static_layers(self, sample, layers, patch_radius=150):
        h, w = self.bev_shape[:2]
        V = self.view
        M_inv = np.array(sample['pose_inverse'])
        S = np.array([ 
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])

        box_coords = (sample['pose'][0][-1] - patch_radius, sample['pose'][1][-1] - patch_radius,
                      sample['pose'][0][-1] + patch_radius, sample['pose'][1][-1] + patch_radius)
        records_in_patch = self.nusc_map.get_records_in_patch(box_coords, layers, 'intersect')

        result = list()

        for layer in layers:
            render = np.zeros((h, w), dtype=np.uint8)

            for r in records_in_patch[layer]:
                polygon_token = self.nusc_map.get(layer, r)

                if layer == 'drivable_area': polygon_tokens = polygon_token['polygon_tokens']
                else: polygon_tokens = [polygon_token['polygon_token']]

                for p in polygon_tokens:
                    polygon = self.nusc_map.extract_polygon(p)
                    polygon = MultiPolygon([polygon])

                    exteriors = [np.array(poly.exterior.coords).T for poly in polygon.geoms]
                    exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in exteriors]
                    exteriors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in exteriors]
                    exteriors = [V @ S @ M_inv @ p for p in exteriors]
                    exteriors = [p[:2].round().astype(np.int32).T for p in exteriors]

                    cv2.fillPoly(render, exteriors, 1, INTERPOLATION)

                    interiors = [np.array(pi.coords).T for poly in polygon.geoms for pi in poly.interiors]
                    interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=0.0) for p in interiors]
                    interiors = [np.pad(p, ((0, 1), (0, 0)), constant_values=1.0) for p in interiors]
                    interiors = [V @ S @ M_inv @ p for p in interiors]
                    interiors = [p[:2].round().astype(np.int32).T for p in interiors]

                    cv2.fillPoly(render, interiors, 0, INTERPOLATION)

            result.append(render)

        return 255 * np.stack(result, -1)

    def get_dynamic_layers(self, sample, anns_by_category):
        h, w = self.bev_shape[:2]
        result = list()

        for anns in anns_by_category:
            render = np.zeros((h, w), dtype=np.uint8)

            for p in self.convert_to_box(sample, anns):
                p = p[:2, :4]

                cv2.fillPoly(render, [p.round().astype(np.int32).T], 1, INTERPOLATION)

            result.append(render)

        return 255 * np.stack(result, -1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Raw annotations
        anns_dynamic = self.get_annotations_by_category(sample, DYNAMIC)
        anns_vehicle = self.get_annotations_by_category(sample, ['vehicle'])[0]

        static = self.get_static_layers(sample, STATIC)                             # 200 200 2
        dividers = self.get_line_layers(sample, DIVIDER)                            # 200 200 2
        dynamic = self.get_dynamic_layers(sample, anns_dynamic)                     # 200 200 8
        bev = np.concatenate((static, dividers, dynamic), -1)                       # 200 200 12

        assert bev.shape[2] == NUM_CLASSES

        # Additional labels for vehicles only.
        aux, visibility = self.get_dynamic_objects(sample, anns_vehicle)

        # Package the data.
        data = Sample(
            view=self.view.tolist(),
            bev=bev,
            aux=aux,
            visibility=visibility,
            **sample
        )
        # if self.transform is not None:
        #     data = self.transform(data) # enter __call__ in class SaveDataTransform in transform.py
        # return data
    
        '''
            1. Get 6 camera images => grd_images
            2. Get satellite map using get_satmap(sample.images[0])
            3. Get intrinsics from data.intrinsic
            4. Get extrinsics from data.extrinsic
            5. Get gt_shift_x, gt_shift_y, theta

        '''

        scene_name = sample['scene']


        # /mnt/workspace: /mnt/workspace/datasets/nuscenes/localization/ + {samples/ | satmap/ }
        SAMPLES_PATH = self.root_dir + 'samples/' 
        SATMAP_PATH  = self.root_dir + 'satmap/' + scene_name + '/'
        # SAMPLES_PATH = '/home/goroyeh/nuScene_dataset/samples/'
        # SATMAP_PATH  = '/home/goroyeh/nuScene_dataset/satmap/' + scene_name + '/'


        # load ground-view images
        grd_imgs = torch.tensor([])
        grd_image_names = sample['images']
        for image_filename in grd_image_names:

            tmp = image_filename.split('/')
            image_name = ''.join( tmp[-2]+'/'+tmp[-1] )
            img_full_path = SAMPLES_PATH + scene_name + '/' + image_name
            # print(f'img_full_path: {img_full_path}')

            with Image.open(img_full_path, 'r') as GrdImg:
                grd_img = GrdImg.convert('RGB')
                if self.grdimage_transform is not None:
                    grd_img = self.grdimage_transform(grd_img)
            grd_imgs = torch.cat([grd_imgs, grd_img.unsqueeze(0)], dim=0)

        intrinsics = torch.FloatTensor(sample['intrinsics'])
        extrinsics = torch.FloatTensor(sample['extrinsics'])

        # Calculate meter_per_pixel
        init_lat, init_lon = MAP_ORIGINS[sample['location']]
        _, _, lat, lon =  getLatLongfromPose(init_lat, init_lon, sample['pose_for_satmap'], "MATLAB")
        meter_per_pixel = utils.get_meter_per_pixel(lat, self.zoom_level)
        # img_name = grd_image_names[0].split("/")[-1]
        # print(f'{img_name} meter_per_pixel: {meter_per_pixel}')

        sat_map_filename= SATMAP_PATH + get_satmap_name(grd_image_names[1])
        if not os.path.exists(sat_map_filename):
            print(f'satmap {sat_map_filename} not exist! Either wrong filename or need to download from google map.')
            # print(f'sample["pose"] {sample["pose"]}')
            download_satmap(init_lat, init_lon, lat, lon, sat_map_filename, self.zoom_level)

        with Image.open(sat_map_filename, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')
            
        heading = sample['yaw']
        heading = torch.from_numpy(np.asarray(heading))

        # print("heading = ", heading)
        # print("sat_map deg = ", -heading / np.pi * 180)
        
        # save_image(sat_map, 'sat_map_origin.png')
        sat_map.save(f'sat_map_origin_{idx}.png')
        # print(sat_map_filename, idx)
        sat_rot = sat_map.rotate(-heading / np.pi * 180 + 90) # +90 is to make vehicle direction facing up

        # Currently no need since both sat and bev are centered
        # sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
        #                                   (1, 0, utils.CameraGPS_shift_left[0] / meter_per_pixel,
        #                                    0, 1, utils.CameraGPS_shift_left[1] / meter_per_pixel),
        #                                   resample=Image.BILINEAR)
        # sat_align_cam = sat_rot
        # the homography is defined on: from target pixel to source pixel
        # now east direction is the real vehicle heading direction

        gt_shift_x = np.random.uniform(-1, 1) * self.shift_range_meters_lat / meter_per_pixel  # shift range is in terms of meters
        gt_shift_y = np.random.uniform(-1, 1) * self.shift_range_meters_lon / meter_per_pixel  # shift range is in terms of meters
        gt_theta = np.random.uniform(-1, 1) * self.rotation_range

        ADDING_GT_TRANSFORM = True # Set this to False if you want to check unshifted dataset
        if not ADDING_GT_TRANSFORM:
            # randomly generate shift
            gt_shift_x *= 0  # --> right as positive, parallel to the heading direction
            gt_shift_y *= 0  # --> up as positive, vertical to the heading direction
            gt_theta *= 0  # --> counter-clockwise as positive

        sat_rand_shift = \
            sat_rot.transform(
                sat_rot.size, Image.AFFINE,
                (1, 0, gt_shift_x,
                0, 1, gt_shift_y),
                resample=Image.BILINEAR)

        sat_rand_shift_rand_rot = \
            sat_rand_shift.rotate(gt_theta)
        
        sat_map =TF.center_crop(sat_rand_shift_rand_rot, utils.SatMap_process_sidelength)

        # transform (resize here)
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        save_image(sat_map, f'sat_map_{idx}.png')

        # print("sat_map.shape = ", sat_map.shape) # (3, 512, 512)
        # print("grd_imgs.shape = ", grd_imgs.shape) # (6, 3, 224, 480)
        sample_name = sample['scene'] + '-' + str(idx)

        return sat_map, grd_imgs, intrinsics, extrinsics, \
               torch.tensor(-gt_shift_x, dtype=torch.float32).reshape(1), \
               torch.tensor(gt_shift_y, dtype=torch.float32).reshape(1), \
               torch.tensor(gt_theta, dtype=torch.float32).reshape(1), \
               meter_per_pixel, \
               sample_name
               


def load_train_data(GrdImg_H, GrdImg_W, version, dataset_dir, labels_dir, loader_config, shift_range_lat, shift_range_lon, rotation_range, root_dir, zoom_level):
    
    SatMap_process_sidelength = utils.get_process_satmap_sidelength()

    satmap_transform = transforms.Compose([
        transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
        transforms.ToTensor(),
    ])

    print(f'GrdImg_H: {GrdImg_H}')
    print(f'GrdImg_W: {GrdImg_W}')    
    
    Grd_h = GrdImg_H 
    Grd_w = GrdImg_W

    grdimage_transform = transforms.Compose([
        transforms.Resize(size=[Grd_h, Grd_w]),
        transforms.ToTensor(),
    ])
    
    
    train_loader = get_split_data(
                             version,
                             dataset_dir, #="/home/goroyeh/nuScene_dataset/media/datasets/nuscenes",
                             labels_dir, #="/home/goroyeh/nuScene_dataset/media/datasets/cvt_labels_nuscenes", 
                             (satmap_transform, grdimage_transform),
                             shift_range_lat, shift_range_lon, rotation_range,
                             root_dir,
                             zoom_level,
                             loader_config,
                             split='train', 
                             loader=True, 
                             shuffle=False)



    return train_loader


def load_val_data(GrdImg_H, GrdImg_W, version, dataset_dir, labels_dir, loader_config, shift_range_lat, shift_range_lon, rotation_range, root_dir, zoom_level):
        
        SatMap_process_sidelength = utils.get_process_satmap_sidelength()

        satmap_transform = transforms.Compose([
            transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
            transforms.ToTensor(),
        ])

        Grd_h = GrdImg_H
        Grd_w = GrdImg_W

        grdimage_transform = transforms.Compose([
            transforms.Resize(size=[Grd_h, Grd_w]),
            transforms.ToTensor(),
        ])
            
        val_loader = get_split_data(
                             version,
                             dataset_dir,
                             labels_dir, 
                             (satmap_transform, grdimage_transform),
                             shift_range_lat, shift_range_lon, rotation_range,
                             root_dir,
                             zoom_level,
                             loader_config,
                             split='val', 
                             loader=True, 
                             shuffle=False)
        return val_loader


def download_satmap(init_lat, init_lon, lat, long, satmap_filename, zoom_level=18):
    # ------- Query satmap from Google Map -------- #
    gmd = GoogleMapDownloader(lat, long, zoom_level, GoogleMapsLayers.SATELLITE)
    img = gmd.generateImage()
    # save images to disk
    print(f'saving {satmap_filename}...')
    img.save(satmap_filename)    

