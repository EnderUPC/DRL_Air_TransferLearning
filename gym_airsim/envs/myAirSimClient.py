import numpy as np
from operator import itemgetter
import time
import math
import cv2
from pylab import array, uint8 
from PIL import Image


from AirSimClient import *

from types import *
from utils import *
from client import *
from pfm import *
import airsim

import random
from reMap import *

client = airsim.MultirotorClient()

class myAirSimClient(MultirotorClient):

    def __init__(self):        

        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)
        
        self.enableApiControl(True,"Drone1")
        self.armDisarm(True,"Drone1")
        #self.enableApiControl(True,"Drone2")
        #self.armDisarm(True,"Drone2")
        #self.enableApiControl(True,"Drone3")
        #self.armDisarm(True,"Drone3")
        #self.enableApiControl(True,"Drone4")
        #self.armDisarm(True,"Drone4")
        self.home_pos = self.simGetGroundTruthKinematics("Drone1").position
        self.home_ori = self.simGetGroundTruthKinematics("Drone1").orientation
        
        '''
        DRONES=["Drone1","Drone2","Drone3","Drone4"]        
        
        for i in DRONES:
            
            self.enableApiControl(True,i)
            self.armDisarm(True,i)
            self.home_pos = self.simGetGroundTruthKinematics(i).position
            self.home_ori = self.simGetGroundTruthKinematics(i).orientation
        '''
        
        self.minx = -5
        self.maxx = 150
        self.miny = -70
        self.maxy = 20
        
        self.z = -4
        
    def movement(self, speed_x, speed_y, speed_z, duration,vehicle_name=''):
        
        #self.moveToPositionAsync(random.randint(-10,10),random.randint(-4,4),self.z, 2, vehicle_name="Drone2")
        #self.moveToPositionAsync(random.randint(0,10),random.randint(-4,4),self.z, 2, vehicle_name="Drone3")
        #self.moveToPositionAsync(random.randint(0,10),random.randint(-4,4),self.z, 2, vehicle_name="Drone4")

        
        #self.moveToPositionAsync(random.randint(0,10),random.randint(-4,4),self.z, 1, vehicle_name="Drone2")
        #self.moveToPositionAsync(random.randint(0,10),random.randint(-4,4),self.z, 1, vehicle_name="Drone3")
        #self.moveToPositionAsync(random.randint(0,10),random.randint(-4,4),self.z, 1, vehicle_name="Drone4")
        
        pitch, roll, yaw  = airsim.to_eularian_angles(client.simGetVehiclePose(vehicle_name="").orientation)
        vel = client.simGetGroundTruthKinematics(vehicle_name="").linear_velocity
        drivetrain = DrivetrainType.ForwardOnly
        yaw_mode = YawMode(is_rate= False, yaw_or_rate = 0)

        self.moveByVelocityAsync(vx = vel.x_val + speed_x,
                            vy = vel.y_val + speed_y,
                            vz = 0,
                            duration = duration,
                            drivetrain = drivetrain,
                            yaw_mode = yaw_mode,vehicle_name="")
        

    
    def take_action(self, action,vehicle_name=''):
        
		 #check if copter is on level cause sometimes he goes up without a reason

        #client.takeoffAsync(vehicle_name="Drone2")


        start = time.time()
        duration = 1 
        
        
        outside = self.geofence(self.minx, self.maxx, 
                                self.miny, self.maxy)
        
        if action == 0:
            
            self.movement(0.5, 0, 0, duration,vehicle_name="")
    
        elif action == 1:
         
            self.movement(-0.5, 0, 0, duration,vehicle_name="")
                
        elif action == 2:
            
            self.movement(0, 0.5, 0, duration,vehicle_name="")
            
                
        elif action == 3:
                    
            self.movement(0, -0.5, 0, duration,vehicle_name="")
                
            
        elif action == 4: # NEW ACTION OPTION------------------------------------> NO MODIFICATION ON CURRENT VELOCITY OF DRONE
                    
            self.movement(0, 0, 0, duration,vehicle_name="")
        
        
        while duration > time.time() - start:
                if self.simGetCollisionInfo(vehicle_name="").has_collided == True:
                    return True
                
                if outside == True:
                    return True
                
        return False
    
    def geofence(self, minx, maxx, miny, maxy,vehicle_name=''):
        
        outside = False
        
        if (self.simGetGroundTruthKinematics(vehicle_name="").position.x_val < minx) or (self.simGetGroundTruthKinematics(vehicle_name="").position.x_val > maxx):
                    return True
        if (self.simGetGroundTruthKinematics(vehicle_name="").position.y_val < miny) or (self.simGetGroundTruthKinematics(vehicle_name="").position.y_val > maxy):
                    return True
                
        return outside
    
    def arrived(self,vehicle_name=''):
        
        landed = self.moveToZAsync(0, 1,vehicle_name="").join()
    
        if landed == True:
            return landed
        
        if (self.simGetGroundTruthKinematics(vehicle_name="").position.z_val > -1):
            return True

    def mapVelocity(self,vehicle_name=''):
        
        vel = client.simGetGroundTruthKinematics(vehicle_name="").linear_velocity
        
        velocity = np.array([vel.x_val, vel.y_val])
        
        return velocity
    
    def mapGeofence(self,vehicle_name=''):
               
        xpos = self.simGetGroundTruthKinematics(vehicle_name="").position.x_val
        ypos = self.simGetGroundTruthKinematics(vehicle_name="").position.y_val
        
        geox1 = self.maxx - xpos
        geox2 = self.minx - xpos
        geoy1 = self.maxy - ypos
        geoy2 = self.miny - ypos

        
        geofence = np.array([geox1, geox2, geoy1, geoy2])
        
        return geofence
    
    def mapDistance(self, goal,vehicle_name=''):
               
        pos = self.simGetGroundTruthKinematics(vehicle_name="").position
        
        xdistance = (goal[0] - (pos.x_val))
        ydistance = (goal[1] - (pos.y_val))

        meandistance = np.sqrt(np.power((goal[0] -pos.x_val),2) + np.power((goal[1] - pos.y_val),2))
        
        distances = np.array([xdistance, ydistance, meandistance])
        
        return distances
    
    def AE(self, goal, vehicle_name=''):
                
        pos = self.simGetGroundTruthKinematics(vehicle_name="").position
        
        xdistance = (goal[0] - (pos.x_val))
        ydistance = (goal[1] - (pos.y_val))
        zdistance = (0 - (pos.z_val))
        
        #elevation=math.atan((zdistance)/np.sqrt(np.power(xdistance,2)+np.power(ydistance,2)))
        #elevation=((math.degrees(elevation) - 90) % 360) - 90
        
        elevation=math.atan2((zdistance),np.sqrt(np.power(xdistance,2)+np.power(ydistance,2)))
        elevation=(math.degrees(elevation))
            
        pitch, roll, yaw  = airsim.to_eularian_angles(client.simGetVehiclePose(vehicle_name="").orientation)
        yaw = math.degrees(yaw) 
        
        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)
        
        #track=((track - math.pi) % 2*math.pi) - math.pi
        track=((math.degrees(track) - 180) % 360) - 180
        
        #track=reMap(track,180,-180,1,-1)
                        
        AE = np.array([track, elevation])
        
        return AE 

        
    
    def getScreenDepthVis(self):

        responses = self.simGetImages([airsim.ImageRequest(0, airsim.ImageType.DepthPerspective,True,False)],vehicle_name="Drone1")
        
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        
        if img1d.size == responses[0].height * responses[0].width:
            # reshape image

            img1d = 255/np.maximum(np.ones(img1d.size), img1d)
            img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
            
            
            image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))
            
            factor = 10
            maxIntensity = 255.0 # depends on dtype of image data
            
            
            # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
            newImage1 = (maxIntensity)*(image/maxIntensity)**factor
            newImage1 = array(newImage1,dtype=uint8)
            
            
            small = cv2.resize(newImage1, (0,0), fx=0.39, fy=0.38)
            
           
            cut = small[20:40,:]
        
            return cut
        
        else:
            #image is empty
            
            #imagedummy = np.zeros((20, 100), dtype=np.uint8)
            
            self.AirSim_reset()
            



    def AirSim_reset(self):
	
        self.reset()
        time.sleep(1)
        
        self.enableApiControl(True,"Drone1")
        self.armDisarm(True,"Drone1")
        #self.enableApiControl(True,"Drone2")
        #self.armDisarm(True,"Drone2")
        #self.enableApiControl(True,"Drone3")
        #self.armDisarm(True,"Drone3")
        #self.enableApiControl(True,"Drone4")
        #self.armDisarm(True,"Drone4")        
        time.sleep(1)
        
        '''
        DRONES=["Drone1","Drone2","Drone3","Drone4"]
        
        for i in DRONES:
            
            self.enableApiControl(True,i)
            self.armDisarm(True,i)
        
        time.sleep(1)
                     
        
        #DRONES.remove("Drone1")
        
        for i in DRONES:
            
            self.moveToZAsync(self.z, 5,vehicle_name=i)

        time.sleep(1)

        '''
        
        self.moveToZAsync(self.z, 3,vehicle_name="Drone1")
        #self.moveToZAsync(self.z, 3,vehicle_name="Drone2")
        #self.moveToZAsync(self.z, 3,vehicle_name="Drone3")
        #self.moveToZAsync(self.z, 3,vehicle_name="Drone4")

        
        time.sleep(1)

        self.moveByVelocityZAsync(0, 0, 0, 3, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(False, random.uniform(-180,180)),vehicle_name="Drone1").join()
        
        time.sleep(1)

        self.moveToZAsync(self.z, 3,vehicle_name="Drone1").join()
        
        time.sleep(1)

 
