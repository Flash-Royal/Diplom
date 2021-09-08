import numpy as np

def cat(size):
    mass = np.zeros((size,size))
    for i in range(len(mass)):
        mass[i,i] = 1
    return mass

class PG():
    def __init__(self, action, alpha, gamma):
        self.gamma = gamma
        self.alpha = alpha
        self.action = action
        self.probActions = []
        self.x = cat(3)
        self.state = np.array(self.x[:1]).reshape(1,3)
        self.theta = np.array([1 for x in range(3)])
        self.actions = []
        # print(self.theta)


    def zip(self, vector):
        newVector = []
        for i, vec in enumerate(vector):
            if i % 2 == 1 and i < 6:
               newVector.append([vector[i-1], vector[i]])
        if newVector[1][0] == 0 or newVector[1][1] == 0:
            newVector[1][0] = 0
            newVector[1][1] = 0
        return newVector

    def relu(self, vector):
        max = 800
        max2 = 400
        for i, vec in enumerate(vector):
            if i == 2 or i == 3:
                vector[i] = 1 if vec > max else 0
            else:
                vector[i] = 1 if vec > max2 else 0
        return vector

    # def firstDense(self):
        # print(np.dot(self.theta, np.array(self.relu(self.zip(self.state))).T))
        # return np.dot(self.theta, np.array(self.relu(self.zip(self.state))).T)

    def softmax(self, vector):
        return np.exp(vector) / sum(np.exp(vector))

    def negativeSoftmax(self, vector):
        num = np.zeros(len(vector))
        num[:] = 1
        return num - vector

    def prob(self):
        # print(np.dot(self.firstDense(),self.x)[0])
        probs = []
        # print(self.state)
        # print(self.zip(self.relu(self.state)))
        for i, el in enumerate(self.zip(self.relu(self.state))):
            # print(self.theta)
            probs.append(self.theta[i] * el[0] + self.theta[i] * el[1])
            # print(self.theta[i])
        # print(probs, "---------", self.softmax(probs))
        return self.softmax(probs)

    def setState(self, state):
        self.state = state

    def randAction(self):
        prob = self.prob()
        act = np.random.choice(self.action, p = prob)
        self.actions.append(self.action.index(act))
        # uno = [1 for x in prob]
        # print(prob)
        print(prob)
        self.probActions.append(prob)
        return act

    def discardR(self, t, rewards):
        T = len(rewards)
        G = 0
        for k in range(t+1, T):
            G += np.power(self.gamma, k-t-1) * rewards[k]
        return G

    def recount(self, rewards):
        J = [0 for i in range(len(self.theta))]
        for i, act in enumerate(self.probActions):
            G = self.discardR(i,rewards)
            # print(self.x[:,self.actions[i]], "=------=", np.dot(self.x, act))
            grad = self.x[:,self.actions[i]] - np.dot(self.x, act)
            # print(grad)
            # print(grad)
            # print(np.dot(self.x, act))
            # -(G)
            J += np.array(grad*G)
            # print(J)
        # J = J / len(self.probActions)
        # print(J)
        # print(self.probActions)
        # print("J = ", J)
        self.probActions = []
        self.actions = []
        self.theta = self.theta + J * self.alpha
        # print(self.theta)
        s = str(self.theta)
        file = open("thetas.txt", 'a')
        # print(s)
        s += '\n'
        file.write(s)
        file.close()
