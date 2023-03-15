#!/usr/bin/python
# GoogleMapDownloader.py
# Created by Hayden Eskriett [http://eskriett.com]
#
# A script which when given a longitude, latitude and zoom level downloads a
# high resolution google map
# Find the associated blog post at: http://blog.eskriett.com/2013/07/19/downloading-google-maps/

import csv
from math import log
import urllib.request
from PIL import Image
import os
import math
import sys

NUM_IMAGES = 10
TILE_SIZE = 1280

SATIMG_INTERVALS = [(101, 199), (5011, 5199)]
# SATIMG_INTERVALS = [(101, 102), (5011, 5012)]


class GoogleMapsLayers:
    ROADMAP = "roadmap"
    TERRAIN = "terrain"
    SATELLITE = "satellite"
    HYBRID = "hybrid"


class GoogleMapDownloader:
    """
        A class which generates high resolution google maps images given
        a longitude, latitude and zoom level
    """

    def __init__(self, lat, lng, zoom=12, layer=GoogleMapsLayers.ROADMAP):
        """
            GoogleMapDownloader Constructor
            Args:
                lat:    The latitude of the location required
                lng:    The longitude of the location required
                zoom:   The zoom level of the location required, ranges from 0 - 23
                        defaults to 12
        """
        self._lat = lat
        self._lng = lng
        self._zoom = zoom
        self._layer = layer

    def project(self):

        siny = math.sin(self._lat * math.pi / 180)
        siny = math.min(math.max(-0.9999), 0.9999)
        return TILE_SIZE * (0.5 + self._lng / 360), TILE_SIZE * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))

    def getXY(self):
        """
            Generates an X,Y tile coordinate based on the latitude, longitude
            and zoom level
            Returns:    An X,Y tile coordinate
        """

        tile_size = TILE_SIZE

        # Use a left shift to get the power of 2
        # i.e. a zoom level of 2 will have 2^2 = 4 tiles
        numTiles = 1 << self._zoom

        # Find the x_point given the longitude
        point_x = (tile_size / 2 + self._lng * tile_size /
                   360.0) * numTiles // tile_size

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calulate the y coorindate
        point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(
            tile_size / (2 * math.pi))) * numTiles // tile_size

        return int(point_x), int(point_y)

    def generateImage(self, **kwargs):
        """
            Generates an image by stitching a number of google map tiles together.

            Args:
                start_x:        The top-left x-tile coordinate
                start_y:        The top-left y-tile coordinate
                tile_width:     The number of tiles wide the image should be -
                                defaults to 5
                tile_height:    The number of tiles high the image should be -
                                defaults to 5
            Returns:
                A high-resolution Goole Map image.
        """

        start_x = kwargs.get('start_x', None)
        start_y = kwargs.get('start_y', None)
        tile_width = kwargs.get('tile_width', 1)
        tile_height = kwargs.get('tile_height', 1)

        # Check that we have x and y tile coordinates
        if start_x == None or start_y == None:
            start_x, start_y = self.getXY()

        # Determine the size of the image
        width, height = TILE_SIZE * tile_width, TILE_SIZE * tile_height

        # Create a new image of the size require
        map_img = Image.new('RGB', (width, height))

        for x in range(0, tile_width):
            for y in range(0, tile_height):
                print("lat = ", self._lat, ", lng = ", self._lng)
                # url = f'https://mt0.google.com/vt?lyrs={self._layer}&x=' + str(start_x + x) + '&y=' + str(start_y + y) + '&z=' + str(
                # self._zoom)
                # Key from goroyeh56@gmail.com
                API_KEY='AIzaSyCINQSR91iBJVrH7CXhNU-wBU6mJWtyIzk'
                # API_KEY = "AIzaSyA7WherFxvKa_f3PLnh1pwZo4KWSdGyQmA" # key from leetkt
                url = f'https://maps.googleapis.com/maps/api/staticmap?center={self._lat},{self._lng}&maptype={self._layer}&zoom={self._zoom}&scale=2&size=640x640&key={API_KEY}'
                # url = f'https://maps.googleapis.com/maps/api/staticmap?v=beta&center={self._lat:.6f},{self._lng:.6f}&maptype={self._layer}&zoom={self._zoom}&scale=1&size=640x640&key=AIzaSyA7WherFxvKa_f3PLnh1pwZo4KWSdGyQmA'
                # url = f'https://maps.googleapis.com/maps/api/vt?v=beta&mapId=1&x=1&y=0&maptype={self._layer}&zoom={self._zoom}&scale=1&size=640x640&key=AIzaSyA7WherFxvKa_f3PLnh1pwZo4KWSdGyQmA'
                # print("url = ", url)
                # url = f'https://maps.googleapis.com/maps/api/staticmap?center=51.477222,0&maptype=satellite&zoom={self._zoom}&size=640x640&key=AIzaSyA7WherFxvKa_f3PLnh1pwZo4KWSdGyQmA'

                # print("start_x = ", start_x, ", start_y = ", start_y)
                # url = f'https://maps.googleapis.com/maps/api/staticmap?x=100&y=100&maptype=satellite&zoom={self._zoom}&key=AIzaSyA7WherFxvKa_f3PLnh1pwZo4KWSdGyQmA'

                current_tile = str(x) + '-' + str(y)
                urllib.request.urlretrieve(url, current_tile)

                im = Image.open(current_tile)
                map_img.paste(im, (x * TILE_SIZE, y * TILE_SIZE))

                os.remove(current_tile)

        return map_img


# import required module


def GetLatLong(filepath):
    f = open(filepath, "r")
    line = f.readline().split(" ")
    lat, lng = float(line[0]), float(line[1])
    return lat, lng


'''
How to get (lat, long) information from each nuscene image?
Steps:
1. Get the (lat, long) of the map origin:

    boston-seaport: [42.336849169438615, -71.05785369873047]
    singapore-onenorth: [1.2882100868743724, 103.78475189208984]
    singapore-hollandvillage: [1.2993652317780957, 103.78217697143555]
    singapore-queenstown: [1.2782562240223188, 103.76741409301758]

2. For each scene, iterate each 'frame' of this scene
3. Use the 'ego_pose' of each frame 
4. Get the transformation matrix from (translation, rotation)
5. Get the (lat, long) of ego_vehicle at this frame (timestep)

6. Query googlemap for this satellite map at timestep k for this scene.


ego_pose
Ego vehicle pose at a particular timestamp. 
Given with respect to global coordinate system of the log's map. 
The ego_pose is the output of a lidar map-based localization algorithm described in our paper. 
The localization is 2-dimensional in the x-y plane.

ego_pose {
   "token":                   <str> -- Unique record identifier.
   "translation":             <float> [3] -- Coordinate system origin in meters: x, y, z. Note that z is always 0.
   "rotation":                <float> [4] -- Coordinate system orientation as quaternion: w, x, y, z.
   "timestamp":               <int> -- Unix time stamp.
}

Need to know initial ego_vehicle_pose: 


Goal:
Stores the satellite map in :
nuscenes/
    satmap/


Training images:
nuScene_dataset/
    media/
        datasets/
            nuscenes/
                samples/
                    CAM_BACK/
                    CAM_BACK_LEFT/
                    CAM_BACK_RIGHT/
                    CAM_FRONT/
                    CAM_FRONT_LEFT/
                    CAM_FRONT_RIGHT/
e.g. Inside CAM_BACK/
    - n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg
    Save the corresponding satellite map to be:
    - n008-2018-05-21-11-06-59-0400__satmap__1526915243037570.jpg


TODO 1:
Organized Samples into each scene:
nuScenes_dataset/
    samples/
        scene-0001
            CAM_BACK/
                n008-2018-05-21-11-06-59-0400__CAM_BACK__1526915243037570.jpg
                ...
            CAM_BACK_LEFT/
            CAM_BACK_RIGHT/
            CAM_FRONT/
            CAM_FRONT_LEFT/
            CAM_FRONT_RIGHT/           

    satmaps/
        scene-0001
            n008-2018-05-21-11-06-59-0400__satmap__1526915243037570.jpg
            ...
        scene-0002


We download US keyframes, so their origin are all:
boston-seaport: [42.336849169438615, -71.05785369873047]
INIT_LAT = 42.336849169438615
INIT_LONG=  -71.05785369873047
'''

from nuscenes.nuscenes import NuScenes
import torch
import numpy as np

# Input: The origin (lat, long), translation(x, y)
# Output: corresponding (lat, long) at this timestamp(give the translation x,y)
def meter2latlon(lat, lon, x, y):
    r = 6378137 # equatorial radius
    flatten = 1/298257 # flattening
    E2 = flatten * (2- flatten)
    m = r * np.pi/180  
    coslat = np.cos(lat * np.pi/180)
    w2 = 1/(1-E2 *(1-coslat*coslat))
    w = np.sqrt(w2)
    kx = m * w * coslat
    ky = m * w * w2 * (1-E2)
    lon += x / kx 
    lat -= y / ky
    
    return lat, lon 

def getLatLongfromSceneIdx(INIT_LAT, INIT_LONG, poses, idx):
    pose = poses[idx]
    # rotation = torch.FloatTensor(pose['rotation'])
    translation = torch.FloatTensor(pose['translation'])  
    x, y, _ = translation  
    print(f'pose (x,y): ({x},{y})')
    # Covert (x,y) to lat,long
    lat, long = meter2latlon(INIT_LAT, INIT_LONG, x,y)  
    return lat, long  

def get_satmap_name(image_name):
    x = ''.join(image_name.split("/")[2:]).split("__")
    x[1] = "_satmap_"
    satmap_filename = ''.join(x)
    return satmap_filename

INIT_LAT = 42.336849169438615
INIT_LONG=  -71.05785369873047

def main():

    if len(sys.argv) < 2:
        print("Error format.\npython3 getSatImg.py <zoom level>")
        return

    print("zoom level: " + str(sys.argv[1]))

    # Path: path to /v1.0-mini/ or /v1.0-trainval
    NUSCENES_DATASET_PATH = '/home/goroyeh/nuScene_dataset/media/datasets/nuscenes'
    nusc = NuScenes(version='v1.0-mini', dataroot=NUSCENES_DATASET_PATH, verbose=True)

    # TODO: This path name should be changed
    nuScene_dataset_PATH = '/home/goroyeh/nuScene_dataset'

    sensors = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    # start, stop indices for getting ego_poses for each scene
    start = 0
    stop = 0
    ego_poses_list = []
    # Iterate each scene
    for scene in nusc.scene:
        scene_name = scene['name']
        print(f'{scene_name} has {scene["nbr_samples"]} samples')
        num_samples = scene["nbr_samples"]

        # Get ego_poses for this scene
        print(f'Get poses for {scene["name"]}')
        stop += scene['nbr_samples']-1
        ego_poses = nusc.ego_pose[start : stop] # a list of k/v pairs
        ego_poses_list.append(nusc.ego_pose[start : stop])
        start = stop+1
        # mkdir satmap/scene_name/
        satmap_scene_path = nuScene_dataset_PATH + '/satmap/' + scene_name
        if not os.path.exists(satmap_scene_path):
            print(f'creating directory: {satmap_scene_path}')
            os.mkdir(satmap_scene_path)
        print(f'Extracing {scene_name} satellite_maps...')

        # --- Set up the first sample --- #
        sample_token = scene['first_sample_token']
        sample = nusc.get('sample', sample_token)
        sample_data = sample['data']

        samples_scene_path = nuScene_dataset_PATH+ '/samples/' + scene_name
        if not os.path.exists(samples_scene_path):
            print(f'creating directory: {samples_scene_path}')
            os.mkdir(samples_scene_path)

        # --- Iterate over samples to setup images and get satmap's name... --- #
        for i in range(num_samples):    
            for sensor in sensors:
                cam_data = nusc.get('sample_data', sample_data[sensor])
                image_name = cam_data['filename']

                # Get images name
                image_name = cam_data['filename']
                print(f'\nimage_name: {image_name}')

                # mkdir samples/scene_name/sensor
                scene_sensor_path = samples_scene_path + sensor
                if not os.path.exists(scene_sensor_path):
                    os.mkdir(scene_sensor_path)
                # Copy this image from original path to new path

                # Get satellite name from this image 
                satmap_name = get_satmap_name(image_name)  
                print(f'satmap_name: {satmap_name}')       
                satmap_name = satmap_scene_path + satmap_name
                if not os.path.exists(satmap_name):  
                    # Get (lat, long) given this ego_pose
                    lat, long =  getLatLongfromSceneIdx(INIT_LAT, INIT_LONG, ego_poses, i)
    
                    # ------- Query satmap from Google Map -------- #
                    gmd = GoogleMapDownloader(lat, long, int(sys.argv[1]), GoogleMapsLayers.SATELLITE)
                    print("The tile coordinates are {}".format(gmd.getXY()))
                    img = gmd.generateImage()
                    # save images to disk
                    
                    img.save(satmap_name)            

            
            # Update sample_token, sample, and sample_data
            sample_token = sample['next']
            sample = nusc.get('sample', sample_token)
            sample_data = sample['data']
                




        # # Get ego_poses for this scene
        # print(f'Get poses for {nusc.scene[i]["name"]}')
        # stop += nusc.scene[i]['nbr_samples']-1
        # ego_poses = nusc.ego_pose[start : stop] # a list of k/v pairs
        # ego_poses_list.append(nusc.ego_pose[start : stop])
        # start = stop+1

        # for pose in ego_poses:
        #     rotation = torch.FloatTensor(pose['rotation'])
        #     translation = torch.FloatTensor(pose['translation'])  
        #     x, y, _ = translation  
        #     print(f'pose (x,y): ({x},{y})')
        #     # Covert (x,y) to lat,long
        #     lat, long = meter2latlon(INIT_LAT, INIT_LONG, x,y)



if __name__ == '__main__':
    main()

