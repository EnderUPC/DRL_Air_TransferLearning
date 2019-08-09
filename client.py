from __future__ import print_function

from types import *
from utils import *
from pfm import *
import airsim

import msgpackrpc #install as admin: pip install msgpack-rpc-python
import numpy as np #pip install numpy
import msgpack
import time
import math
import logging


class MsgpackMixin:
    def __repr__(self):
        from pprint import pformat
        return "<" + type(self).__name__ + "> " + pformat(vars(self), indent=4, width=1)

    def to_msgpack(self, *args, **kwargs):
        return self.__dict__

    @classmethod
    def from_msgpack(cls, encoded):
        obj = cls()
        #obj.__dict__ = {k.decode('utf-8'): (from_msgpack(v.__class__, v) if hasattr(v, "__dict__") else v) for k, v in encoded.items()}
        obj.__dict__ = { k : (v if not isinstance(v, dict) else getattr(getattr(obj, k).__class__, "from_msgpack")(v)) for k, v in encoded.items()}
        #return cls(**msgpack.unpack(encoded))
        return obj


class ImageType:    
    Scene = 0
    DepthPlanner = 1
    DepthPerspective = 2
    DepthVis = 3
    DisparityNormalized = 4
    Segmentation = 5
    SurfaceNormals = 6
    Infrared = 7

class DrivetrainType:
    MaxDegreeOfFreedom = 0
    ForwardOnly = 1
    
class LandedState:
    Landed = 0
    Flying = 1

class WeatherParameter:
    Rain = 0
    Roadwetness = 1
    Snow = 2
    RoadSnow = 3
    MapleLeaf = 4
    RoadLeaf = 5
    Dust = 6
    Fog = 7
    Enabled = 8

class Vector3r(MsgpackMixin):
    x_val = 0.0
    y_val = 0.0
    z_val = 0.0

    def __init__(self, x_val = 0.0, y_val = 0.0, z_val = 0.0):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val

    @staticmethod
    def nanVector3r():
        return Vector3r(np.nan, np.nan, np.nan)

    def __add__(self, other):
        return Vector3r(self.x_val + other.x_val, self.y_val + other.y_val, self.z_val + other.z_val)

    def __sub__(self, other):
        return Vector3r(self.x_val - other.x_val, self.y_val - other.y_val, self.z_val - other.z_val)

    def __truediv__(self, other):
        if type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Vector3r( self.x_val / other, self.y_val / other, self.z_val / other)
        else: 
            raise TypeError('unsupported operand type(s) for /: %s and %s' % ( str(type(self)), str(type(other))) )

    def __mul__(self, other):
        if type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Vector3r(self.x_val*other, self.y_val*other, self.z_val)
        else: 
            raise TypeError('unsupported operand type(s) for *: %s and %s' % ( str(type(self)), str(type(other))) )

    def dot(self, other):
        if type(self) == type(other):
            return self.x_val*other.x_val + self.y_val*other.y_val + self.z_val*other.z_val
        else:
            raise TypeError('unsupported operand type(s) for \'dot\': %s and %s' % ( str(type(self)), str(type(other))) )

    def cross(self, other):
        if type(self) == type(other):
            cross_product = np.cross(self.to_numpy_array(), other.to_numpy_array())
            return Vector3r(cross_product[0], cross_product[1], cross_product[2])
        else:
            raise TypeError('unsupported operand type(s) for \'cross\': %s and %s' % ( str(type(self)), str(type(other))) )

    def get_length(self):
        return ( self.x_val**2 + self.y_val**2 + self.z_val**2 )**0.5

    def distance_to(self, other):
        return ( (self.x_val-other.x_val)**2 + (self.y_val-other.y_val)**2 + (self.z_val-other.z_val)**2 )**0.5

    def to_Quaternionr(self):
        return Quaternionr(self.x_val, self.y_val, self.z_val, 0)

    def to_numpy_array(self):
        return np.array([self.x_val, self.y_val, self.z_val], dtype=np.float32)


class Quaternionr(MsgpackMixin):
    w_val = 0.0
    x_val = 0.0
    y_val = 0.0
    z_val = 0.0

    def __init__(self, x_val = 0.0, y_val = 0.0, z_val = 0.0, w_val = 1.0):
        self.x_val = x_val
        self.y_val = y_val
        self.z_val = z_val
        self.w_val = w_val

    @staticmethod
    def nanQuaternionr():
        return Quaternionr(np.nan, np.nan, np.nan, np.nan)

    def __add__(self, other):
        if type(self) == type(other):
            return Quaternionr( self.x_val+other.x_val, self.y_val+other.y_val, self.z_val+other.z_val, self.w_val+other.w_val )
        else:
            raise TypeError('unsupported operand type(s) for +: %s and %s' % ( str(type(self)), str(type(other))) )

    def __mul__(self, other):
        if type(self) == type(other):
            t, x, y, z = self.w_val, self.x_val, self.y_val, self.z_val
            a, b, c, d = other.w_val, other.x_val, other.y_val, other.z_val
            return Quaternionr( w_val = a*t - b*x - c*y - d*z,
                                x_val = b*t + a*x + d*y - c*z,
                                y_val = c*t + a*y + b*z - d*x,
                                z_val = d*t + z*a + c*x - b*y)
        else:
            raise TypeError('unsupported operand type(s) for *: %s and %s' % ( str(type(self)), str(type(other))) )

    def __truediv__(self, other): 
        if type(other) == type(self): 
            return self * other.inverse()
        elif type(other) in [int, float] + np.sctypes['int'] + np.sctypes['uint'] + np.sctypes['float']:
            return Quaternionr( self.x_val / other, self.y_val / other, self.z_val / other, self.w_val / other)
        else: 
            raise TypeError('unsupported operand type(s) for /: %s and %s' % ( str(type(self)), str(type(other))) )

    def dot(self, other):
        if type(self) == type(other):
            return self.x_val*other.x_val + self.y_val*other.y_val + self.z_val*other.z_val + self.w_val*other.w_val
        else:
            raise TypeError('unsupported operand type(s) for \'dot\': %s and %s' % ( str(type(self)), str(type(other))) )

    def cross(self, other):
        if type(self) == type(other):
            return (self * other - other * self) / 2
        else:
            raise TypeError('unsupported operand type(s) for \'cross\': %s and %s' % ( str(type(self)), str(type(other))) )

    def outer_product(self, other):
        if type(self) == type(other):
            return ( self.inverse()*other - other.inverse()*self ) / 2
        else:
            raise TypeError('unsupported operand type(s) for \'outer_product\': %s and %s' % ( str(type(self)), str(type(other))) )

    def rotate(self, other):
        if type(self) == type(other):
            if other.get_length() == 1:
                return other * self * other.inverse()
            else:
                raise ValueError('length of the other Quaternionr must be 1')
        else:
            raise TypeError('unsupported operand type(s) for \'rotate\': %s and %s' % ( str(type(self)), str(type(other))) )        

    def conjugate(self):
        return Quaternionr(-self.x_val, -self.y_val, -self.z_val, self.w_val)

    def star(self):
        return self.conjugate()

    def inverse(self):
        return self.star() / self.dot(self)

    def sgn(self):
        return self/self.get_length()

    def get_length(self):
        return ( self.x_val**2 + self.y_val**2 + self.z_val**2 + self.w_val**2 )**0.5

    def to_numpy_array(self):
        return np.array([self.x_val, self.y_val, self.z_val, self.w_val], dtype=np.float32)


class Pose(MsgpackMixin):
    position = Vector3r()
    orientation = Quaternionr()

    def __init__(self, position_val = Vector3r(), orientation_val = Quaternionr()):
        self.position = position_val
        self.orientation = orientation_val

    @staticmethod
    def nanPose():
        return Pose(Vector3r.nanVector3r(), Quaternionr.nanQuaternionr())


class CollisionInfo(MsgpackMixin):
    has_collided = False
    normal = Vector3r()
    impact_point = Vector3r()
    position = Vector3r()
    penetration_depth = 0.0
    time_stamp = 0.0
    object_name = ""
    object_id = -1

class GeoPoint(MsgpackMixin):
    latitude = 0.0
    longitude = 0.0
    altitude = 0.0

class YawMode(MsgpackMixin):
    is_rate = True
    yaw_or_rate = 0.0
    def __init__(self, is_rate = True, yaw_or_rate = 0.0):
        self.is_rate = is_rate
        self.yaw_or_rate = yaw_or_rate

class RCData(MsgpackMixin):
    timestamp = 0
    pitch, roll, throttle, yaw = (0.0,)*4 #init 4 variable to 0.0
    switch1, switch2, switch3, switch4 = (0,)*4
    switch5, switch6, switch7, switch8 = (0,)*4
    is_initialized = False
    is_valid = False
    def __init__(self, timestamp = 0, pitch = 0.0, roll = 0.0, throttle = 0.0, yaw = 0.0, switch1 = 0,
                 switch2 = 0, switch3 = 0, switch4 = 0, switch5 = 0, switch6 = 0, switch7 = 0, switch8 = 0, is_initialized = False, is_valid = False):
        self.timestamp = timestamp
        self.pitch = pitch 
        self.roll = roll
        self.throttle = throttle 
        self.yaw = yaw 
        self.switch1 = switch1 
        self.switch2 = switch2 
        self.switch3 = switch3 
        self.switch4 = switch4 
        self.switch5 = switch5
        self.switch6 = switch6 
        self.switch7 = switch7 
        self.switch8 = switch8 
        self.is_initialized = is_initialized
        self.is_valid = is_valid

class ImageRequest(MsgpackMixin):
    camera_name = '0'
    image_type = ImageType.Scene
    pixels_as_float = False
    compress = False

    def __init__(self, camera_name, image_type, pixels_as_float = False, compress = True):
        # todo: in future remove str(), it's only for compatibility to pre v1.2
        self.camera_name = str(camera_name)
        self.image_type = image_type
        self.pixels_as_float = pixels_as_float
        self.compress = compress


class ImageResponse(MsgpackMixin):
    image_data_uint8 = np.uint8(0)
    image_data_float = 0.0
    camera_position = Vector3r()
    camera_orientation = Quaternionr()
    time_stamp = np.uint64(0)
    message = ''
    pixels_as_float = 0.0
    compress = True
    width = 0
    height = 0
    image_type = ImageType.Scene

class CarControls(MsgpackMixin):
    throttle = 0.0
    steering = 0.0
    brake = 0.0
    handbrake = False
    is_manual_gear = False
    manual_gear = 0
    gear_immediate = True

    def __init__(self, throttle = 0, steering = 0, brake = 0, 
        handbrake = False, is_manual_gear = False, manual_gear = 0, gear_immediate = True):
        self.throttle = throttle
        self.steering = steering
        self.brake = brake
        self.handbrake = handbrake
        self.is_manual_gear = is_manual_gear
        self.manual_gear = manual_gear
        self.gear_immediate = gear_immediate


    def set_throttle(self, throttle_val, forward):
        if (forward):
            is_manual_gear = False
            manual_gear = 0
            throttle = abs(throttle_val)
        else:
            is_manual_gear = False
            manual_gear = -1
            throttle = - abs(throttle_val)

class KinematicsState(MsgpackMixin):
    position = Vector3r()
    orientation = Quaternionr()
    linear_velocity = Vector3r()
    angular_velocity = Vector3r()
    linear_acceleration = Vector3r()
    angular_acceleration = Vector3r()

class EnvironmentState(MsgpackMixin):
    position = Vector3r()
    geo_point = GeoPoint()
    gravity = Vector3r()
    air_pressure = 0.0
    temperature = 0.0
    air_density = 0.0

class CarState(MsgpackMixin):
    speed = 0.0
    gear = 0
    rpm = 0.0
    maxrpm = 0.0
    handbrake = False
    collision = CollisionInfo();
    kinematics_estimated = KinematicsState()
    timestamp = np.uint64(0)

class MultirotorState(MsgpackMixin):
    collision = CollisionInfo();
    kinematics_estimated = KinematicsState()
    gps_location = GeoPoint()
    timestamp = np.uint64(0)
    landed_state = LandedState.Landed
    rc_data = RCData()

class ProjectionMatrix(MsgpackMixin):
    matrix = []

class CameraInfo(MsgpackMixin):
    pose = Pose()
    fov = -1
    proj_mat = ProjectionMatrix()

class LidarData(MsgpackMixin):
    point_cloud = 0.0
    time_stamp = np.uint64(0)
    pose = Pose()


class VehicleClient:
    def __init__(self, ip = "", port = 41451, timeout_value = 3600):
        if (ip == ""):
            ip = "127.0.0.1"
        self.client = msgpackrpc.Client(msgpackrpc.Address(ip, port), timeout = timeout_value, pack_encoding = 'utf-8', unpack_encoding = 'utf-8')
        
    # -----------------------------------  Common vehicle APIs ---------------------------------------------
    def reset(self):
        self.client.call('reset')

    def ping(self):
        return self.client.call('ping')

    def getClientVersion(self):
        return 1 # sync with C++ client
    def getServerVersion(self):
        return self.client.call('getServerVersion')
    def getMinRequiredServerVersion(self):
        return 1 # sync with C++ client
    def getMinRequiredClientVersion(self):
        return self.client.call('getMinRequiredClientVersion')

    # basic flight control
    def enableApiControl(self, is_enabled, vehicle_name = ''):
        return self.client.call('enableApiControl', is_enabled, vehicle_name)
    def isApiControlEnabled(self, vehicle_name = ''):
        return self.client.call('isApiControlEnabled', vehicle_name)
    def armDisarm(self, arm, vehicle_name = ''):
        return self.client.call('armDisarm', arm, vehicle_name)
 
    def simPause(self, is_paused):
        self.client.call('simPause', is_paused)
    def simIsPause(self):
        return self.client.call("simIsPaused")
    def simContinueForTime(self, seconds):
        self.client.call('simContinueForTime', seconds)

    def getHomeGeoPoint(self, vehicle_name = ''):
        return GeoPoint.from_msgpack(self.client.call('getHomeGeoPoint', vehicle_name))

    def confirmConnection(self):
        if self.ping():
            print("Connected!")
        else:
             print("Ping returned false!")
        server_ver = self.getServerVersion()
        client_ver = self.getClientVersion()
        server_min_ver = self.getMinRequiredServerVersion()
        client_min_ver = self.getMinRequiredClientVersion()
    
        ver_info = "Client Ver:" + str(client_ver) + " (Min Req: " + str(client_min_ver) + \
              "), Server Ver:" + str(server_ver) + " (Min Req: " + str(server_min_ver) + ")"

        if server_ver < server_min_ver:
            print(ver_info, file=sys.stderr)
            print("AirSim server is of older version and not supported by this client. Please upgrade!")
        elif client_ver < client_min_ver:
            print(ver_info, file=sys.stderr)
            print("AirSim client is of older version and not supported by this server. Please upgrade!")
        else:
            print(ver_info)
        print('')

    # time-of-day control
    def simSetTimeOfDay(self, is_enabled, start_datetime = "", is_start_datetime_dst = False, celestial_clock_speed = 1, update_interval_secs = 60, move_sun = True):
        return self.client.call('simSetTimeOfDay', is_enabled, start_datetime, is_start_datetime_dst, celestial_clock_speed, update_interval_secs, move_sun)

    # weather
    def simEnableWeather(self, enable):
        return self.client.call('simEnableWeather', enable)
    def simSetWeatherParameter(self, param, val):
        return self.client.call('simSetWeatherParameter', param, val)

    # camera control
    # simGetImage returns compressed png in array of bytes
    # image_type uses one of the ImageType members
    def simGetImage(self, camera_name, image_type, vehicle_name = ''):
        # todo: in future remove below, it's only for compatibility to pre v1.2
        camera_name = str(camera_name)

        # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
        result = self.client.call('simGetImage', camera_name, image_type, vehicle_name)
        if (result == "" or result == "\0"):
            return None
        return result

    # camera control
    # simGetImage returns compressed png in array of bytes
    # image_type uses one of the ImageType members
    def simGetImages(self, requests, vehicle_name = ''):
        responses_raw = self.client.call('simGetImages', requests, vehicle_name)
        return [ImageResponse.from_msgpack(response_raw) for response_raw in responses_raw]

    def simGetCollisionInfo(self, vehicle_name = ''):
        return CollisionInfo.from_msgpack(self.client.call('simGetCollisionInfo', vehicle_name))

    def simSetVehiclePose(self, pose, ignore_collison, vehicle_name = ''):
        self.client.call('simSetVehiclePose', pose, ignore_collison, vehicle_name)
    def simGetVehiclePose(self, vehicle_name = ''):
        pose = self.client.call('simGetVehiclePose', vehicle_name)
        return Pose.from_msgpack(pose)
    def simGetObjectPose(self, object_name):
        pose = self.client.call('simGetObjectPose', object_name)
        return Pose.from_msgpack(pose)
    def simSetObjectPose(self, object_name, pose, teleport = True):
        return self.client.call('simSetObjectPose', object_name, pose, teleport)

    def simSetSegmentationObjectID(self, mesh_name, object_id, is_name_regex = False):
        return self.client.call('simSetSegmentationObjectID', mesh_name, object_id, is_name_regex)
    def simGetSegmentationObjectID(self, mesh_name):
        return self.client.call('simGetSegmentationObjectID', mesh_name)
    def simPrintLogMessage(self, message, message_param = "", severity = 0):
        return self.client.call('simPrintLogMessage', message, message_param, severity)

    def simGetCameraInfo(self, camera_name, vehicle_name = ''):
        # TODO: below str() conversion is only needed for legacy reason and should be removed in future
        return CameraInfo.from_msgpack(self.client.call('simGetCameraInfo', str(camera_name), vehicle_name))
    def simSetCameraOrientation(self, camera_name, orientation, vehicle_name = ''):
        # TODO: below str() conversion is only needed for legacy reason and should be removed in future
        self.client.call('simSetCameraOrientation', str(camera_name), orientation, vehicle_name)

    def simGetGroundTruthKinematics(self, vehicle_name = ''):
        kinematics_state = self.client.call('simGetGroundTruthKinematics', vehicle_name)
        return KinematicsState.from_msgpack(kinematics_state)
    simGetGroundTruthKinematics.__annotations__ = {'return': KinematicsState}
    def simGetGroundTruthEnvironment(self, vehicle_name = ''):
        env_state = self.client.call('simGetGroundTruthEnvironment', vehicle_name)
        return EnvironmentState.from_msgpack(env_state)
    simGetGroundTruthEnvironment.__annotations__ = {'return': EnvironmentState}

    # lidar APIs
    def getLidarData(self, lidar_name = '', vehicle_name = ''):
        return LidarData.from_msgpack(self.client.call('getLidarData', lidar_name, vehicle_name))

    #----------- APIs to control ACharacter in scene ----------/
    def simCharSetFaceExpression(self, expression_name, value, character_name = ""):
        self.client.call('simCharSetFaceExpression', expression_name, value, character_name)
    def simCharGetFaceExpression(self, expression_name, character_name = ""):
        return self.client.call('simCharGetFaceExpression', expression_name, character_name)
    def simCharGetAvailableFaceExpressions(self):
        return self.client.call('simCharGetAvailableFaceExpressions')
    def simCharSetSkinDarkness(self, value, character_name = ""):
        self.client.call('simCharSetSkinDarkness', value, character_name)
    def simCharGetSkinDarkness(self, character_name = ""):
        return self.client.call('simCharGetSkinDarkness', character_name)
    def simCharSetSkinAgeing(self, value, character_name = ""):
        self.client.call('simCharSetSkinAgeing', value, character_name)
    def simCharGetSkinAgeing(self, character_name = ""):
        return self.client.call('simCharGetSkinAgeing', character_name)
    def simCharSetHeadRotation(self, q, character_name = ""):
        self.client.call('simCharSetHeadRotation', q, character_name)
    def simCharGetHeadRotation(self, character_name = ""):
        return self.client.call('simCharGetHeadRotation', character_name)
    def simCharSetBonePose(self, bone_name, pose, character_name = ""):
        self.client.call('simCharSetBonePose', bone_name, pose, character_name)
    def simCharGetBonePose(self, bone_name, character_name = ""):
        return self.client.call('simCharGetBonePose', bone_name, character_name)
    def simCharResetBonePose(self, bone_name, character_name = ""):
        self.client.call('simCharResetBonePose', bone_name, character_name)
    def simCharSetFacePreset(self, preset_name, value, character_name = ""):
        self.client.call('simCharSetFacePreset', preset_name, value, character_name)
    def simCharSetFacePresets(self, presets, character_name = ""):
        self.client.call('simSetFacePresets', presets, character_name)
    def simCharSetBonePoses(self, poses, character_name = ""):
        self.client.call('simSetBonePoses', poses, character_name)
    def simCharGetBonePoses(self, bone_names, character_name = ""):
        return self.client.call('simGetBonePoses', bone_names, character_name)

    def cancelLastTask():
        self.client.call('cancelLastTask')
    def waitOnLastTask(timeout_sec = float('nan')):
        return self.client.call('waitOnLastTask', timeout_sec)

    # legacy handling
    # TODO: remove below legacy wrappers in future major releases
    upgrade_api_help = "\nPlease see https://github.com/Microsoft/AirSim/blob/master/docs/upgrade_apis.md for more info."
    def simGetPose(self):
        logging.warning("simGetPose API is renamed to simGetVehiclePose. Please update your code." + self.upgrade_api_help)
        return self.simGetVehiclePose()
    def simSetPose(self, pose, ignore_collison):
        logging.warning("simSetPose API is renamed to simSetVehiclePose. Please update your code." + self.upgrade_api_help)
        return self.simSetVehiclePose(pose, ignore_collison)
    def getCollisionInfo(self):
        logging.warning("getCollisionInfo API is renamed to simGetCollisionInfo. Please update your code." + self.upgrade_api_help)
        return self.simGetCollisionInfo()
    def getCameraInfo(self, camera_id):
        logging.warning("getCameraInfo API is renamed to simGetCameraInfo. Please update your code." + self.upgrade_api_help)
        return self.simGetCameraInfo(camera_id)
    def setCameraOrientation(self, camera_id, orientation):
        logging.warning("setCameraOrientation API is renamed to simSetCameraOrientation. Please update your code." + self.upgrade_api_help)
        return self.simSetCameraOrientation(camera_id, orientation)
    def getPosition(self):
        logging.warning("getPosition API is deprecated. For ground-truth please use simGetGroundTruthKinematics() API." + self.upgrade_api_help)
        return self.simGetGroundTruthKinematics().position
    def getVelocity(self):
        logging.warning("getVelocity API is deprecated. For ground-truth please use simGetGroundTruthKinematics() API." + self.upgrade_api_help)
        return self.simGetGroundTruthKinematics().linear_velocity
    def getOrientation(self):
        logging.warning("getOrientation API is deprecated. For ground-truth please use simGetGroundTruthKinematics() API." + self.upgrade_api_help)
        return self.simGetGroundTruthKinematics().orientation
    def getLandedState(self):
        raise Exception("getLandedState API is deprecated. Please use getMultirotorState() API")
    def getGpsLocation(self):
        logging.warning("getGpsLocation API is deprecated. For ground-truth please use simGetGroundTruthKinematics() API." + self.upgrade_api_help)
        return self.simGetGroundTruthEnvironment().geo_point
    def takeoff(self, max_wait_seconds = 15):
        raise Exception("takeoff API is deprecated. Please use takeoffAsync() API." + self.upgrade_api_help)
    def land(self, max_wait_seconds = 60):
        raise Exception("land API is deprecated. Please use landAsync() API." + self.upgrade_api_help)
    def goHome(self):
        raise Exception("goHome API is deprecated. Please use goHomeAsync() API." + self.upgrade_api_help)
    def hover(self):
        raise Exception("hover API is deprecated. Please use hoverAsync() API." + self.upgrade_api_help)
    def moveByAngleZ(self, pitch, roll, z, yaw, duration):
        raise Exception("moveByAngleZ API is deprecated. Please use moveByAngleZAsync() API." + self.upgrade_api_help)
    def moveByAngleThrottle(self, pitch, roll, throttle, yaw_rate, duration):
        raise Exception("moveByAngleThrottle API is deprecated. Please use moveByAngleThrottleAsync() API." + self.upgrade_api_help)
    def moveByVelocity(self, vx, vy, vz, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode()):
        raise Exception("moveByVelocity API is deprecated. Please use moveByVelocityAsync() API." + self.upgrade_api_help)
    def moveByVelocityZ(self, vx, vy, z, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode()):
        raise Exception("moveByVelocityZ API is deprecated. Please use moveByVelocityZAsync() API." + self.upgrade_api_help)
    def moveOnPath(self, path, velocity, max_wait_seconds = 60, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1):
        raise Exception("moveOnPath API is deprecated. Please use moveOnPathAsync() API." + self.upgrade_api_help)
    def moveToZ(self, z, velocity, max_wait_seconds = 60, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1):
        raise Exception("moveToZ API is deprecated. Please use moveToZAsync() API." + self.upgrade_api_help)
    def moveToPosition(self, x, y, z, velocity, max_wait_seconds = 60, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1):
        raise Exception("moveToPosition API is deprecated. Please use moveToPositionAsync() API." + self.upgrade_api_help)
    def moveByManual(self, vx_max, vy_max, z_min, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode()):
        raise Exception("moveByManual API is deprecated. Please use moveByManualAsync() API." + self.upgrade_api_help)
    def rotateToYaw(self, yaw, max_wait_seconds = 60, margin = 5):
        raise Exception("rotateToYaw API is deprecated. Please use rotateToYawAsync() API." + self.upgrade_api_help)
    def rotateByYawRate(self, yaw_rate, duration):
        raise Exception("rotateByYawRate API is deprecated. Please use rotateByYawRateAsync() API." + self.upgrade_api_help)
    def setRCData(self, rcdata = RCData()):
        raise Exception("setRCData API is deprecated. Please use moveByRC() API." + self.upgrade_api_help)

# -----------------------------------  Multirotor APIs ---------------------------------------------
class MultirotorClient(VehicleClient, object):
    def __init__(self, ip = "", port = 41451, timeout_value = 3600):
        super(MultirotorClient, self).__init__(ip, port, timeout_value)

    def takeoffAsync(self, timeout_sec = 20, vehicle_name = ''):
        return self.client.call_async('takeoff', timeout_sec, vehicle_name)  
    def landAsync(self, timeout_sec = 60, vehicle_name = ''):
        return self.client.call_async('land', timeout_sec, vehicle_name)   
    def goHomeAsync(self, timeout_sec = 3e+38, vehicle_name = ''):
        return self.client.call_async('goHome', timeout_sec, vehicle_name)

    # APIs for control
    def moveByAngleZAsync(self, pitch, roll, z, yaw, duration, vehicle_name = ''):
        return self.client.call_async('moveByAngleZ', pitch, roll, z, yaw, duration, vehicle_name)
    def moveByAngleThrottleAsync(self, pitch, roll, throttle, yaw_rate, duration, vehicle_name = ''):
        return self.client.call_async('moveByAngleThrottle', pitch, roll, throttle, yaw_rate, duration, vehicle_name)
    def moveByVelocityAsync(self, vx, vy, vz, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), vehicle_name = ''):
        return self.client.call_async('moveByVelocity', vx, vy, vz, duration, drivetrain, yaw_mode, vehicle_name)
    def moveByVelocityZAsync(self, vx, vy, z, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), vehicle_name = ''):
        return self.client.call_async('moveByVelocityZ', vx, vy, z, duration, drivetrain, yaw_mode, vehicle_name)
    def moveOnPathAsync(self, path, velocity, timeout_sec = 3e+38, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), 
        lookahead = -1, adaptive_lookahead = 1, vehicle_name = ''):
        return self.client.call_async('moveOnPath', path, velocity, timeout_sec, drivetrain, yaw_mode, lookahead, adaptive_lookahead, vehicle_name)
    def moveToPositionAsync(self, x, y, z, velocity, timeout_sec = 3e+38, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), 
        lookahead = -1, adaptive_lookahead = 1, vehicle_name = ''):
        return self.client.call_async('moveToPosition', x, y, z, velocity, timeout_sec, drivetrain, yaw_mode, lookahead, adaptive_lookahead, vehicle_name)
    def moveToZAsync(self, z, velocity, timeout_sec = 3e+38, yaw_mode = YawMode(), lookahead = -1, adaptive_lookahead = 1, vehicle_name = ''):
        return self.client.call_async('moveToZ', z, velocity, timeout_sec, yaw_mode, lookahead, adaptive_lookahead, vehicle_name)
    def moveByManualAsync(self, vx_max, vy_max, z_min, duration, drivetrain = DrivetrainType.MaxDegreeOfFreedom, yaw_mode = YawMode(), vehicle_name = ''):
        """Read current RC state and use it to control the vehicles. 

        Parameters sets up the constraints on velocity and minimum altitude while flying. If RC state is detected to violate these constraints
        then that RC state would be ignored.

        :param vx_max: max velocity allowed in x direction
        :param vy_max: max velocity allowed in y direction
        :param vz_max: max velocity allowed in z direction
        :param z_min: min z allowed for vehicle position
        :param duration: after this duration vehicle would switch back to non-manual mode
        :param drivetrain: when ForwardOnly, vehicle rotates itself so that its front is always facing the direction of travel. If MaxDegreeOfFreedom then it doesn't do that (crab-like movement)
        :param yaw_mode: Specifies if vehicle should face at given angle (is_rate=False) or should be rotating around its axis at given rate (is_rate=True)
        """
        return self.client.call_async('moveByManual', vx_max, vy_max, z_min, duration, drivetrain, yaw_mode, vehicle_name)
    def rotateToYawAsync(self, yaw, timeout_sec = 3e+38, margin = 5, vehicle_name = ''):
        return self.client.call_async('rotateToYaw', yaw, timeout_sec, margin, vehicle_name)
    def rotateByYawRateAsync(self, yaw_rate, duration, vehicle_name = ''):
        return self.client.call_async('rotateByYawRate', yaw_rate, duration, vehicle_name)
    def hoverAsync(self, vehicle_name = ''):
        return self.client.call_async('hover', vehicle_name)

    def moveByRC(self, rcdata = RCData(), vehicle_name = ''):
        return self.client.call('moveByRC', rcdata, vehicle_name)
        
    # query vehicle state
    def getMultirotorState(self, vehicle_name = ''):
        return MultirotorState.from_msgpack(self.client.call('getMultirotorState', vehicle_name))
    getMultirotorState.__annotations__ = {'return': MultirotorState}


# -----------------------------------  Car APIs ---------------------------------------------
class CarClient(VehicleClient, object):
    def __init__(self, ip = "", port = 41451, timeout_value = 3600):
        super(CarClient, self).__init__(ip, port, timeout_value)

    def setCarControls(self, controls, vehicle_name = ''):
        self.client.call('setCarControls', controls, vehicle_name)

    def getCarState(self, vehicle_name = ''):
        state_raw = self.client.call('getCarState', vehicle_name)
        return CarState.from_msgpack(state_raw)
