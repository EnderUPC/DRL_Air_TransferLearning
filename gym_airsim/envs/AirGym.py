import logging
import numpy as np

import gym
import sys
from gym import spaces
from gym.utils import seeding
from gym.spaces import Box
from gym.spaces.box import Box

from gym_airsim.envs.myAirSimClient import *
        
from AirSimClient import *

logger = logging.getLogger(__name__)

from types import *
from utils import *
from client import *
from pfm import *
import airsim

class AirSimEnv(gym.Env):

    airgym = None

        
    def __init__(self):
        
        #self.cum_reward = 0.0
        #self.discount = 0.8 #check others
        self.simage = np.zeros((20, 100), dtype=np.uint8)
        self.svelocity = np.zeros((2,), dtype=np.float32)
        self.sdistance = np.zeros((3,), dtype=np.float32)
        self.sgeofence = np.zeros((4,), dtype=np.float32)
        

        self.sAE  = np.zeros((2,), dtype=np.float32)  
        
        
        self.action_space = spaces.Discrete(5)
		
        self.goal = 	[112,10]
        
        self.episodeN = 0
        self.stepN = 0 
        
        self.allLogs = { 'reward':[0] }
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]


        self.seed()
        
        global airgym
        airgym = myAirSimClient()
        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def state(self):
        
        return self.simage, self.svelocity, self.sdistance, self.sgeofence, self.sAE
        
    def computeReward(self, now):
	
      
        distance_now = np.sqrt(np.power((self.goal[0]-now.x_val),2) + np.power((self.goal[1]-now.y_val),2) )
        
        distance_before = self.allLogs['distance'][-1]
              
        r = -1
        
        r = r + (distance_before - distance_now)
            
        return r, distance_now
		
    
    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.addToLog('action', action)
        
        self.stepN += 1

        collided = airgym.take_action(action,"Drone1")
        
        now = airgym.simGetGroundTruthKinematics(vehicle_name="Drone1").position
        
        colli_info_D1 = airgym.simGetCollisionInfo(vehicle_name="Drone1")
        
        #colli_info_D2 = airgym.simGetCollisionInfo(vehicle_name="Drone2")
        #colli_info_D3 = airgym.simGetCollisionInfo(vehicle_name="Drone3")
        #colli_info_D4 = airgym.simGetCollisionInfo(vehicle_name="Drone4")

        #colli_info_D1 = airgym.simGetCollisionInfo(vehicle_name="Drone1")
        
        if collided == True:
            
            done = True
            reward = -100.0
            distance = np.sqrt(np.power((self.goal[0]),2) + np.power((self.goal[1]),2))      
            
            
            '''
            if colli_info_D2.object_name == "Drone1" and colli_info_D2.has_collided == True :
                    done = True
                    reward = -10.0
                    distance = np.sqrt(np.power((self.goal[0]),2) + np.power((self.goal[1]),2))
                    print("ACHTUNG! DRONE1 IS INNOCENT")
                    with open("crashreport.txt", "a") as myfile:
                        myfile.write("\n"+"Episode"+ "\t" +str(self.episodeN) + '\t' + "DRONE1 IS INNOCENT")
                                            
                     
            if (colli_info_D2.object_name == "Drone1" and colli_info_D2.has_collided == True) or (colli_info_D3.object_name == "Drone1" and colli_info_D3.has_collided == True) or (colli_info_D4.object_name == "Drone1" and colli_info_D4.has_collided == True) :
                    done = True
                    reward = -10.0
                    distance = np.sqrt(np.power((self.goal[0]),2) + np.power((self.goal[1]),2))
                    print("ACHTUNG! DRONE1 IS INNOCENT")
                    with open("crashreport.txt", "a") as myfile:
                        myfile.write("\n"+"Episode"+ "\t" +str(self.episodeN) + '\t' + "DRONE1 IS INNOCENT")
            
            
            if (colli_info_D2.object_name == "Drone1" and colli_info_D2.has_collided == True) or (colli_info_D3.object_name == "Drone1" and colli_info_D3.has_collided == True) :
                    done = True
                    reward = -10.0
                    distance = np.sqrt(np.power((self.goal[0]),2) + np.power((self.goal[1]),2))
                    print("ACHTUNG! DRONE1 IS INNOCENT")
                    with open("crashreport.txt", "a") as myfile:
                        myfile.write("\n"+"Episode"+ "\t" +str(self.episodeN) + '\t' + "DRONE1 IS INNOCENT" + '\t' + colli_info_D1.object_name)
                        
            else:    
                done = True
                reward = -100.0
                distance = np.sqrt(np.power((self.goal[0]),2) + np.power((self.goal[1]),2))
                print("ACHTUNG! DRONE1 HIT SOMETHING")
                with open("crashreport_HIT.txt", "a") as myfile:
                        myfile.write("\n"+"Episode"+ "\t" +str(self.episodeN) + '\t' + "DRONE1 HIT SOMETHING")
            '''            
            
        else: 
            done = False
            reward, distance = self.computeReward(now)
            
            
        # Youuuuu made it
        if distance < 3:
            landed = airgym.arrived()
            if landed == True:
                done = True
                reward = 100.0
            
                with open("reached.txt", "a") as myfile:
                    myfile.write(str(self.episodeN) + ", ")
            
        self.addToLog('reward', reward)
        rewardSum = np.sum(self.allLogs['reward'])
        #self.cum_reward=self.discount * self.cum_reward + reward
        self.addToLog('distance', distance)  
            
        # Terminate the episode on large cumulative amount penalties, 
        # since drone probably got into an unexpected loop of some sort
        if rewardSum < -300:
            done = True
       

        
        info = {"x_pos" : now.x_val, "y_pos" : now.y_val}
        
        self.simage = airgym.getScreenDepthVis()
        self.svelocity = airgym.mapVelocity("Drone1")
        self.sdistance = airgym.mapDistance(self.goal,"Drone1")
        self.sgeofence = airgym.mapGeofence("Drone1")
        
        self.sAE  = airgym.AE(self.goal, "Drone1")

        sys.stdout.write("\r\x1b[K{}/{}==>reward/rewardSum/distance/Track_A/Elevation_A: {:.1f}/{:.1f}/{:.1f}//{:.1f}/{:.1f}    \t  {:.0f} \t".format(self.episodeN, self.stepN, reward, rewardSum,self.sdistance[-1],self.sAE[0],self.sAE[1], action))
        sys.stdout.flush()
                
        #self.sangle2goal  = airgym.angle2goal(self.goal,"Drone1")
        #self.selevation  = airgym.elevationangle(self.goal, "Drone1")



        
        state = self.state()
        return state, reward, done, info

    def addToLog (self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)
        
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        airgym.AirSim_reset()
        
        totalrewards = np.sum(self.allLogs['reward'])
        with open("rewards.txt", "a") as myfile:
            myfile.write(str(totalrewards) + ", ")
        
       # arr = np.array([[137.5, -48.7], [59.1, -15.1], [-62.3, -7.35], [123, 77.3]])
       # probs = [.25, .25, .25, .25]
        #indicies = np.random.choice(len(arr), 1, p=probs)
        #array = (arr[indicies])
        #list = (array.tolist())
        
        '''
        self.goal = [item for sublist in list for item in sublist]
        '''
        self.goal = 	[112,10]

        self.stepN = 0
        self.episodeN += 1
        
        distance = np.sqrt(np.power((self.goal[0]),2) + np.power((self.goal[1]),2) )
        self.allLogs = { 'reward': [0] }
        self.allLogs['distance'] = [distance]
        self.allLogs['action'] = [1]
        
        
  
        self.simage = airgym.getScreenDepthVis()
        self.svelocity = airgym.mapVelocity("Drone1")
        self.sdistance = airgym.mapDistance(self.goal,"Drone1")
        self.sgeofence = airgym.mapGeofence("Drone1")
        
        #self.sangle2goal  = airgym.angle2goal(self.goal,"Drone1")
        #self.selevation  = airgym.elevationangle(self.goal, "Drone1")

        self.sAE  = airgym.AE(self.goal, "Drone1")
        
        state = self.state()
        

        
        return state
