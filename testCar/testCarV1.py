"""testCar controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Robot, Keyboard
import numpy as np
import GUI
import policy
import os
import shutil
# create the Robot instance.

class MyCar():
    def __init__(self, timestep, speed, alpha = 0.0002, gamma = 0.95):
        self.robot = Robot()
        self.wheelNames = ['left wheel 1','left wheel 2','right wheel 1','right wheel 2']
        self.leftWheels = [self.robot.getDevice(self.wheelNames[i]) for i in range(2)]
        self.rightWheels = [self.robot.getDevice(self.wheelNames[i + 2]) for i in range(2)]
        self.timestep = timestep
        self.alpha = alpha
        self.gamma = gamma
        self.keyboard = Keyboard()
        self.IRSensorsNames = ['dsL', 'ds0', 'ds1', 'ds2', 'ds3', 'dsR']
        self.actions = ['left', 'front', 'right']
        self.IRSensors = [self.robot.getDevice(sensor) for sensor in self.IRSensorsNames]
        self.Actor = policy.PG(self.actions, alpha, gamma)
        self.state = 0
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.gps = self.robot.getDevice("gps")
        self.compass = self.robot.getDevice('compass')
        self.speed = speed
        self.map = GUI.Display()

    def startPos(self):
        self.receiver.enable(self.timestep)
        self.gps.enable(self.timestep)
        self.compass.enable(self.timestep)
        self.keyboard.enable(self.timestep)

        for sensor in self.IRSensors:
            sensor.enable(self.timestep)

        for left, right in zip(self.leftWheels, self.rightWheels):
            left.setPosition(float('inf'))
            left.setVelocity(0.0)
            right.setPosition(float('inf'))
            right.setVelocity(0.0)

    def control(self, key):
        if key == ord('W'):
            leftSpeed = self.speed
            rightSpeed = self.speed
        elif key == ord('S'):
            leftSpeed = -self.speed
            rightSpeed = -self.speed
        elif key == ord('A'):
            leftSpeed = 0
            rightSpeed = self.speed
        elif key == ord('D'):
            leftSpeed = self.speed
            rightSpeed = 0
        else:
            leftSpeed = 0
            rightSpeed = 0

        for left, right in zip(self.leftWheels, self.rightWheels):
            left.setVelocity(leftSpeed)
            right.setVelocity(rightSpeed)

    def finishPos(self):
        self.receiver.disable()
        self.gps.disable()
        self.compass.disable()
        for sensor in self.IRSensors:
            sensor.disable()

        for left, right in zip(self.leftWheels, self.rightWheels):
            left.setPosition(float('inf'))
            left.setVelocity(0.0)
            right.setPosition(float('inf'))
            right.setVelocity(0.0)

    def getSensorsValues(self):
        IRSensorValues = []
        # print(self.gps.getValues())
        for sensor in self.IRSensors:
            IRSensorValues.append(sensor.getValue())
        return IRSensorValues

    def getGPSValues(self):
        gpsVal = self.gps.getValues()
        return gpsVal

    def getAngle(self):
        compassVal = self.compass.getValues()
        rad = np.arctan2(compassVal[0], compassVal[2])
        angle = rad - 1.5708
        return angle

    def chooseAction(self):
        sens = self.getSensorsValues()
        self.Actor.setState(sens)
        # print("sens :", sens)
        return self.Actor.randAction()


    def giveReward(self):
        reward = []
        while self.receiver.getQueueLength() > 0:
            msg = self.receiver.getData().decode("utf-8")
            reward.append(int(msg))
            self.receiver.nextPacket()
        reward.append(-100)
        return reward

    def send(self, msg):
        msg1 = msg.encode("utf-8")
        self.emitter.send(msg1)

    def takeAction(self, action):
        sensors = self.getSensorsValues()
        # print("sensors: ", sensors)
        if sensors[1] < 70 or sensors[2] < 150 or sensors[3] < 150 or sensors[4] < 70:
            state = -1
        else:
            state = 0

        return state

    def speedCar(self, action):
        if action == 'left':
            leftSpeed = 0
            rightSpeed = self.speed
        elif action == "right":
            leftSpeed = self.speed
            rightSpeed = 0
        else:
            leftSpeed = self.speed
            rightSpeed = self.speed

        for left, right in zip(self.leftWheels, self.rightWheels):
            left.setVelocity(leftSpeed)
            right.setVelocity(rightSpeed)

    def cleanFile(self):
        file = open("episodes.txt", 'w')
        file.close()
        file = open("thetas.txt", 'w')
        file.close()
        shutil.rmtree("images")
        os.mkdir("images")

    def train(self, ep = 1000):
        # self.startPos()
        self.cleanFile()
        actions = []
        rewards = []
        for i in range(0, ep):
            print("Epizode {}".format(i))
            self.startPos()
            while self.robot.step(self.timestep) != -1:
                gpsVal = self.getGPSValues()
                key= self.keyboard.getKey()
                # self.control(key)
                action = self.chooseAction()
                self.speedCar(action)
                self.state = self.takeAction(action)
                self.map.draw(gpsVal[0], gpsVal[2], self.getSensorsValues(), 0.07, 0.1, 0.3, self.getAngle(), 100)
                # print(reward)
                actions.append(action)
                if self.state == -1:
                    rewards = self.giveReward()
                    # print(len(rewards), '-----', len(actions))
                    self.map.restart(i)
                    self.Actor.recount(rewards)
                    actions = []
                    self.send("1")
                    s = "round {}: reward {}".format(i, sum(rewards))
                    file = open("episodes.txt", 'a')
                    print(s)
                    s += '\n'
                    file.write(s)
                    file.close()
                    rewards = []
                    self.state = 0
                    self.finishPos()
                    break

    def predict(self):
        action = self.chooseAction()
        self.speedCar(action)

robot = MyCar(64, 2)
robot.train(10000)
