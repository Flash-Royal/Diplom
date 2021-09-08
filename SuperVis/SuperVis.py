"""SuperVis controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor

import numpy as np

timeStep = 32

class SuperVis():
    def __init__(self, robotName, timestep):
        self.supervisor = Supervisor()
        self.robotNode = self.supervisor.getFromDef(robotName)
        self.transField = self.robotNode.getField("translation")
        self.rotField = self.robotNode.getField("rotation")
        self.finishNode = self.supervisor.getFromDef('Finish')
        self.transFinish = self.finishNode.getField("translation")
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(timestep)
        self.pos = self.transField.getSFVec3f()
        
    def getStartPosAndRot(self):
        pos = self.pos
        rot = self.rotField.getSFRotation()
        return pos, rot
    
    def getFinishPos(self):
        pos = self.transFinish.getSFVec3f()
        return pos
    
    def getPos(self):
        newTransField = self.robotNode.getField("translation")
        pos = newTransField.getSFVec3f()
        return pos
    
    def getStartLen(self):
        pos1 = self.pos
        pos2 = self.transFinish.getSFVec3f()
        return self.len(pos1,pos2)
        
    def setStartPos(self, pos, rot):
        self.transField.setSFVec3f(pos)
        self.rotField.setSFRotation(rot)
    
    def reward(self, pos):
        pos1 = pos
        pos2 = self.getFinishPos()
        len = self.len(pos1, pos2)
        # print(self.getStartLen())
        reward = (self.getStartLen() - len)
        return int(reward)
        
    def len(self, pos1, pos2):
        return np.sqrt((pos2[2] - pos1[2])**2 + (pos2[0] - pos1[0])**2)
        
    def send(self, reward):
        msg = str(reward).encode("utf-8")
        self.emitter.send(msg)
   
    def receive(self):
        #print("queue: ", self.receiver.getQueueLength())
        if self.receiver.getQueueLength() > 0:
            msg = self.receiver.getData().decode("utf-8")
            #print ("size = ", self.receiver.getDataSize())
            self.receiver.nextPacket()
            
        else:
            msg = '0'
        # print(msg)
        return msg
        
# print(dir(supervisor.getField))
robotVis = SuperVis("MyRob", timeStep)
pos, rot = robotVis.getStartPosAndRot()

while robotVis.supervisor.step(timeStep) != -1:
    # this is done repeatedly
    if robotVis.receive() == '1':
        robotVis.setStartPos(pos,rot)
    else:
        pos1 = robotVis.getPos()
        reward = robotVis.reward(pos1)
        robotVis.send(reward)
 