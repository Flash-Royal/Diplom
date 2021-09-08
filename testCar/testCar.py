"""testCar controller."""

from controller import Robot
import numpy as np
import map
import os
import shutil
from reinforce_keras import Agent
import time

class MyCar():
    def __init__(self, timestep, speed, alpha = 0.0002, gamma = 0.95):
        self.robot = Robot()
        self.wheelNames = ['left wheel 1','left wheel 2','right wheel 1','right wheel 2']
        self.leftWheels = [self.robot.getDevice(self.wheelNames[i]) for i in range(2)]
        self.rightWheels = [self.robot.getDevice(self.wheelNames[i + 2]) for i in range(2)]
        self.timestep = timestep
        self.alpha = alpha
        self.gamma = gamma
        self.IRSensorsNames = ['dsL', 'ds0', 'ds1', 'ds2', 'ds3', 'dsR']
        self.actions = ['left', 'front', 'right']
        self.IRSensors = [self.robot.getDevice(sensor) for sensor in self.IRSensorsNames]
        self.Agent = Agent(alpha = alpha, input_dims = 6, gamma = gamma, n_actions = 3, layer1_size = 64, layer2_size = 64)
        self.state = 0
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.gps = self.robot.getDevice("gps")
        self.compass = self.robot.getDevice('compass')
        self.speed = speed
        self.map = map.Display()
        self.rewards = []

    def startPos(self):
        self.receiver.enable(self.timestep)
        self.gps.enable(self.timestep)
        self.compass.enable(self.timestep)

        for sensor in self.IRSensors:
            sensor.enable(self.timestep)

        for left, right in zip(self.leftWheels, self.rightWheels):
            left.setPosition(float('inf'))
            left.setVelocity(0.0)
            right.setPosition(float('inf'))
            right.setVelocity(0.0)

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

    def optimizeState(self, vector):
        max = 800
        max2 = 600
        for i, vec in enumerate(vector):
            if i == 2 or i == 3:
                vector[i] = 1 if vec > max else 0
            else:
                vector[i] = 1 if vec > max2 else 0
        return vector

    def chooseAction(self):
        state = self.getSensorsValues()
        state = self.optimizeState(state)
        action = self.Agent.choose_action(state)
        # print("sens :", sens)
        return state, action

    def giveReward(self):
        # print("len = ", self.receiver.getQueueLength())
        if self.receiver.getQueueLength() > 0:
            msg = self.receiver.getData().decode("utf-8")
            reward = int(msg)
            self.receiver.nextPacket()
            self.receiver.nextPacket()
        return reward

    def send(self, msg):
        msg1 = msg.encode("utf-8")
        self.emitter.send(msg1)

    def takeAction(self):
        sensors = self.getSensorsValues()
        # print(sensors)
        # print("sensors: ", sensors)
        if sensors[1] < 70 or sensors[2] < 150 or sensors[3] < 150 or sensors[4] < 70:
            state = -1
        else:
            state = 0

        return state

    def speedCar(self, action):
        if action == 0:
            leftSpeed = 0
            rightSpeed = self.speed
        elif action == 2:
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

    def train(self, ep = 100000):
        self.cleanFile()
        if os.path.exists('model_weights.h5'):
            self.Agent.load_model()
        for i in range(0, ep):
            print("Epizode {}".format(i))
            score = 0
            while self.robot.step(self.timestep) != -1:
                self.startPos()
                gpsVal = self.getGPSValues()
                reward = self.giveReward()
                state, action = self.chooseAction()
                self.speedCar(action)
                self.state = self.takeAction()
                if self.state == -1:
                    reward = -100
                self.Agent.store_transition(state, action, reward)
                score += reward
                self.map.draw(gpsVal[0], gpsVal[2], self.getSensorsValues(), 0.07, 0.1, 0.3, self.getAngle(), 100)
                if self.state == -1:
                    self.rewards.append(score)
                    self.map.restart(i)
                    self.Agent.learn()
                    avgReward = np.mean(self.rewards[-100:])
                    s = "round {}: reward {}, avg reward {}".format(i, score, avgReward)
                    file = open("episodes.txt", 'a')
                    print(s)
                    s += '\n'
                    file.write(s)
                    file.close()
                    rewards = []
                    self.state = 0
                    self.finishPos()
                    self.Agent.save_model()
                    self.send("1")
                    time.sleep(1)
                    break

    def predict(self):
        action = self.chooseAction()
        self.speedCar(action)

robot = MyCar(64, 2)
robot.train(1000)
