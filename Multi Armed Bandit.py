import numpy as np
import matplotlib.pyplot as plt
        
class MultiArmedBandit(object):
    def __init__(self, k, totalSteps, iterations):
        self.k = k                          # number of arms    
        self.totalSteps = totalSteps        # total time steps, here = 1000
        self.iterations = iterations
        
        self.curStep = 0                    # current time step
        self.curReward = 0                  # reward of chosen action
        self.count = np.zeros(k)            # how many times a particular arm is being selected
        self.rewardSum = np.zeros(k)        # Sums number of rewards
        self.Q = np.zeros(k)                # estimated value Q = sum(rewards)/count
        
        
    
    def generateQStar(self, mean = 0, stdev = 1):
        self.mean = mean        # mean and variance for normal distribution, here mean = 0, var = 1
        self.stdev = stdev
        
        self.qstar = np.random.normal(self.mean, self.stdev, self.k)  # actual values of 10 bandits 
        self.maxim = np.argmax(self.qstar)                            # maximum value to be maintained for greedy action
        
        
    def update(self):
        self.count[self.action] += 1
        self.curReward = np.random.normal(self.qstar[self.action],self.stdev)
        self.rewardSum[self.action] += self.curReward
        
        if (self.alpha == 0):
            self.Q[self.action] = self.Q[self.action] + (self.curReward - self.Q[self.action])/self.count[self.action]
        else:
            self.Q[self.action] = self.Q[self.action] + self.alpha*(self.curReward - self.Q[self.action])
        
        
    #epsilon greedy action selection
    def selectAction(self):                             
        randomProb = np.random.uniform(low=0, high=1)   # pick random probability 
        self.action = 0
        
        # explore i.e. select arm randomly
        if randomProb < self.epsilon:                   
            self.action = np.random.choice(self.k)
        
        # exploit i.e. select arm w max reward
        else:                                           
            possibleActions = np.where(self.Q == max(self.Q))[0] 
                   
            # only one action with max value
            if len(possibleActions) == 1:               
                self.action = possibleActions[0] 
                
            # select random action if multiple actions w max value
            else:                                       
                self.action = np.random.choice(possibleActions)
                
        #updating values after action chosen
        self.update()
        
        
    def ucb(self):
        if self.count.min() == 0:
            self.action = np.random.choice(np.flatnonzero(self.count == self.count.min()))
        
        else:
            self.action = np.argmax(self.Q + self.c * np.sqrt(np.divide(np.log(self.curStep),self.count)))
            
        #updating values after action chosen
        self.update()
            
        
    # Reset all variables for next iteration
    def reset(self):
        self.curStep = 0                   
        self.curReward = 0              

        self.count[:] = 0                  
        self.rewardSum[:] = 0
        self.Q[:] = self.initValue
        
        
    def play(self, epsilon = 0, initValue = 0, alpha = 0, c = 0):
        self.epsilon = epsilon
        self.initValue = initValue
        self.alpha = alpha
        self.c = c
        
        #Expected reward values
        self.scoreArr = np.zeros((self.totalSteps, 1))
        
        #Optimal Action values
        self.optArr = np.zeros((self.totalSteps,1))
        
        for _ in range(self.iterations):
            self.generateQStar()
            self.reset()
            
            for step in range(self.totalSteps):
                if (self.c == 0):
                    self.selectAction()
                else:
                    self.ucb()
                    
                self.scoreArr[self.curStep] += self.curReward
                if (self.action == self.maxim):
                    self.optArr[self.curStep] += 1
                self.curStep += 1
                
        #return averages
        scoreAvg = self.scoreArr/self.iterations
        optimlAvg = self.optArr/self.iterations

        return scoreAvg, optimlAvg
    
    def testbedPlot(self):
        
        plt.plot([0,12],[0,0],linestyle='--')
        plt.plot(np.arange(10)+1,self.qstar,'ro',label='$ \ mean \ (\mu$)')
        plt.errorbar(np.arange(10)+1,self.qstar,yerr=np.ones(10),fmt='none', label='Standard deviation $\ (\sigma$)')
        plt.title('10-armed testbed')
        plt.ylim(min(self.qstar)-2, max(self.qstar)+2)
        plt.xlabel('Action')
        plt.ylabel('Reward Distribution')
        plt.legend()
        plt.show()
        
        
    def plot_avgReturn(self,ar,legend,title,colours,ucb_flag = 0):
        self.colours = colours
        self.ar = ar
        self.legend = legend
        self.title = title
        
        #Plotting average rewards vs time steps
        plt.title(title)
        
        if (ucb_flag == 0):
            plt.xlim(0,1000)
            plt.ylim(0,1.5)
            plt.xticks(np.arange(0, 1001, 250))
            plt.yticks(np.arange(0, 1.6, 0.5))
        
        for i in range(len(ar)):
            plt.plot(ar[i],colours[i], label=legend[i],linewidth = '1.2')
        
        plt.ylabel('Average Reward')
        plt.xlabel('Steps')
        plt.legend(legend)
        plt.show()
        
    
    def plot_optimal(self,op,legend,title,colours, ucb_flag = 0):
        self.colours = colours
        self.op = op
        self.legend = legend
        self.title = title
        
        #Plotting average rewards vs time steps
        plt.title(title)
        
        if (ucb_flag == 0):
            plt.xlim(0,1000)
            plt.ylim(0,100)
            plt.xticks(np.arange(0, 1001, 250))
            plt.yticks(np.arange(0, 101, 20))
        
        for i in range(len(op)):
            plt.plot(op[i]*100,colours[i], label=legend[i],linewidth = '1.2')
        
        plt.ylabel('% Optimal Action')
        plt.xlabel('Steps')
        plt.legend(legend)
        plt.show()
        
        
def runQ1():
    bandit = MultiArmedBandit(10, 1000, 2000)
    bandit.generateQStar()
    
    # Plot Reward distribution for 10-arms
    print("Plotting reward distribution for 10-armed bandid problem")
    bandit.testbedPlot()
    
    #Greedy method
    print("Performing Greedy method (Ɛ=0)...")
    avgReward0,optAct0 = bandit.play(0,0,0)

    print("Performing Ɛ-Greedy method (Ɛ=0.1)...")
    avgReward1,optAct1 = bandit.play(0.1,0,0)

    print("Performing Ɛ-Greedy method (Ɛ=0.01)...")
    avgReward2,optAct2 = bandit.play(0.01,0,0)
    
    print("Performing Ɛ-Greedy method (Ɛ=0.2)...")
    avgReward6,optAct6 = bandit.play(0.2,0,0)
    
    print("Performing Ɛ-Greedy method (Ɛ=0.5)...")
    avgReward7,optAct7 = bandit.play(0.5,0,0)
    
    print("Performing Ɛ-Greedy method (Ɛ=0.8)...")
    avgReward8,optAct8 = bandit.play(0.8,0,0)
    
    
    legend = ['Ɛ=0', 'Ɛ=0.1', 'Ɛ=0.01','Ɛ=0.2', 'Ɛ=0.5', 'Ɛ=0.8']
    title = "10-Armed TestBed - Average Rewards"
    ar = [avgReward0,avgReward1,avgReward2,avgReward6,avgReward7,avgReward8]
    colours = ['g','b','r','c','y','0.8']
    bandit.plot_avgReturn(ar,legend,title,colours)
    
    title = "10-Armed TestBed - Average Rewards"
    op = [optAct0,optAct1,optAct2,optAct6,optAct7,optAct8]
    bandit.plot_optimal(op,legend,title,colours)
    
    
    optimisticInitialValue = MultiArmedBandit(10, 1000, 2000)
    print("Performing Optimistic Initial Value Method (Ɛ=0, Q_initial = 5)...")
    avgReward3,optAct3 = optimisticInitialValue.play(0,5,0.1)
    
    print("Performing Optimistic Initial Value Method (Ɛ=0, Q_initial = 10)...")
    avgReward9,optAct9 = optimisticInitialValue.play(0,10,0.1)
    
    print("Performing Epsilon Greedy Method (Ɛ=0, Q_initial = 0)...")
    avgReward10,optAct10 = bandit.play(0,0,0.1)
    
    print("Performing Epsilon Greedy Method (Ɛ=0.1, Q_initial = 0)...")
    avgReward4,optAct4 = bandit.play(0.1,0,0.1)
    
    print("Performing Epsilon Greedy + Optimistic Initial Value Method (Ɛ=0.1, Q_initial = 5)...")
    avgReward11,optAct11 = bandit.play(0.1,5,0.1)
    
    #textbook
    legend = ['Optimistic Greedy', 'Realistic Greedy']
    title = "Optimistic Initial Values v/s Realistic Epsilon Greedy - % Optimal Action"
    op = [optAct3,optAct4]
    colours = ['b','0.8']
    bandit.plot_optimal(op,legend,title,colours)
    
    #tuning
    legend = ['Ɛ=0, Q_initial = 5', 'Ɛ=0.1, Q_initial = 0', 'Ɛ=0, Q_initial = 10','Ɛ=0, Q_initial = 0','Ɛ=0.1, Q_initial = 5']
    title = "Optimistic Initial Values v/s Realistic Epsilon Greedy - % Optimal Action"
    op = [optAct3,optAct4,optAct9,optAct10,optAct11]
    colours = ['b','0.8','c','r','g']
    bandit.plot_optimal(op,legend,title,colours)

    upperConfBound = MultiArmedBandit(10, 1000, 2000)
    print("Performing UCB Method (c = 0.1)...")
    avgReward12,optAct12 = upperConfBound.play(0,0,0,0.1)
    
    print("Performing UCB Method (c = 1)...")
    avgReward13,optAct13 = upperConfBound.play(0,0,0,1)
    
    print("Performing UCB Method (c = 2)...")
    avgReward5,optAct5 = upperConfBound.play(0,0,0,2)
    
    print("Performing UCB Method (c = 3)...")
    avgReward14,optAct14 = upperConfBound.play(0,0,0,3)
    
    #textbook
    legend = ['c = 2', 'Epsilon greedy with Ɛ = 0.1']
    title = "UCB v/s Epsilon Greedy - Average Performance"
    ar = [avgReward5,avgReward1]
    colours = ['b','0.8']
    bandit.plot_avgReturn(ar,legend,title,colours)
    
    title = "UCB v/s Epsilon Greedy - % Optimal Action"
    op = [optAct5,optAct1]
    bandit.plot_optimal(op,legend,title,colours)
    
    #tuning
    legend = ['c = 0.1', 'c = 1', 'c = 2', 'c = 3', 'Epsilon greedy with Ɛ = 0.1']
    title = "UCB v/s Epsilon Greedy - Average Performance"
    ar = [avgReward12,avgReward13,avgReward5,avgReward14,avgReward1]
    colours = ['b','r','g','c','0.8']
    bandit.plot_avgReturn(ar,legend,title,colours,1)
    
    title = "UCB v/s Epsilon Greedy - % Optimal Action"
    op = [optAct12,optAct13,optAct5,optAct14,optAct1]
    bandit.plot_optimal(op,legend,title,colours,1)
        

#****************************************** Main Function ***********************************************
if __name__ == '__main__':
    runQ1()
    