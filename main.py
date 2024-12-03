import math 
from torch import nn
import torch
import numpy as np

'''
RL steps:
==========================================
1. Agent observes the state of the environment (s).
2. Agent takes action (a) based on its policy π on the state s.
3. Agent interact with the environment. A new state is formed.
4. Agent takes further actions based on the observed state.
5. After a trajectory τ of motions, it adjusts its policy based on the total rewards R(τ) received.

to do list:
==========================================
- better visualization of the game and agent's actions 
- run mini-experiments to understand how to implement a policy and value network correctly
- run-mini-simulations to test the reward function
- figure out when to update gradients 
- better gradient computation 

'''
class Policy(nn.Module):
    '''
    neural network that gives an action given a state
    '''
    def __init__(self):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_features=3, out_features=64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

        self.head = nn.Softmax()

    def forward(self, x):
        x = torch.tensor(x)
        x = x.to(torch.float32)
        logits = self.MLP(x)
        return self.head(logits)

class Value(nn.Module):
    '''
    neural network that estimates the expected return/reward at a given state
    '''
    def __init__(self):
        super().__init__()
        self.MLP = nn.Sequential(
            nn.Linear(in_features=3, out_features=32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = torch.tensor(x)
        x = x.to(torch.float32)
        logits = self.MLP(x)
        return logits

class Agent:
    def __init__(self):
        self.policy_network = Policy()
        self.value_network = Value()

    def policy(self, state):
        '''
        The optimal policy achieves optimal value functions
        action = int of set of 4 (1: up, 2: right, 3: down, 4: left)

        state = (x,y), has reached goal 

        input = [list of 3 numbers] e.g.: [1,1,0]

        output = [list of 4 probs, each ranging from 0 to 1]

        action = max(output)

        neural network (state) -> action 
    
        '''
        probs = self.policy_network((state['pos'][0], state['pos'][1], int(state["has_reached_goal"])))
        action = torch.argmax(probs.detach()) + 1 # get the action from index + 1

        return action, torch.log(max(probs))
        
    def value(self, state):
        expected_return = self.value_network((state['pos'][0], state['pos'][1], int(state["has_reached_goal"])))

        return expected_return

class Environment:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.reached_goal = False
        self.goal = (9, 9)
        self.state = {'pos': (self.x, self.y), 'has_reached_goal': self.reached_goal}

        self.map = [[0 for _ in range(10)] for _ in range(10)]
        self.map[9][9] = 'X'
        self.map[0][0] = 'A'

        self.discount = 1
        self.time = 0

    ''' 
    1, 3: up and down
    2, 4: left and right

    pos y + action % 2
    '''
    def update_env(self, action):
        if action == 1:
            self.y += 1
        elif action == 2:
            self.x += 1
        elif action == 3:
            self.y -= 1
        elif action == 4:
            self.x -= 1
        
        if self.state['pos'] == self.goal:
            self.state['has_reached_goal'] = True

    def update_map(self, prev_state, state):
        self.map[prev_state['pos'][0]][prev_state['pos'][1]] = 0 
        self.map[state['pos'][0]][state['pos'][1]] = 'A'

    def is_move_legal(self, action, state):
        if state['pos'][0] == 0 and action == 1:
            return False
        if state['pos'][0] == 9 and action == 3:
            return False
        if state['pos'][1] == 0 and action == 4:
            return False
        if state['pos'][1] == 9 and action == 2:
            return False
        
    def reward(self, state, next_state):
        '''
        computes reward at time step t
        '''
        init_dist = self.distance(state["pos"][0], state["pos"][1], self.goal[0], self.goal[1])
        next_dist = self.distance(next_state["pos"][0], next_state["pos"][1], self.goal[0], self.goal[1])

        return init_dist - next_dist

    def distance(self, x1, y1, x2, y2):
        return math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))

    def render(self):
        '''
        - render the map for each time step t for one game 
        - print the reward
        '''
        for i in range(len(self.map)):
            print(self.map[i], '\n')
        print('\n')

class Game:
    def __init__(self):
        self.env = Environment()
        self.agent = Agent()
        self.trajectories = []
        self.reward_lists = []
        self.expected_return_lists = []
        self.discount  = 1
        self.lr = 0.01
        #self.optimizer_1 = torch.optim.SGD(self.agent.policy_network, lr=self.lr)
        #self.optimizer_2 = torch.optim.SGD(self.agent.value_network.parameters(), lr=self.lr)
        #self.rewards_to_go = 0

    def calc_rewards_to_go(self, rewards, time):
        rewards_to_go = 0
        for t_prime in range(time, len(rewards)):
            rewards_to_go += math.pow(self.discount, t_prime) * rewards[t_prime]
        
        return rewards_to_go
    
    def calc_adv(self, rewards_to_go, expected_value):
        return rewards_to_go-expected_value
    
    def calc_value_loss(self, expected_return, reward):
        MSE = (expected_return - reward)**2
        return MSE
       
    def game_loop(self, update_interval):
        '''
        one epoch 
        '''
        prev_state = self.env.state
        finish_reward = 0

        trajectory = []
        reward_list = []
        expected_return_list = []

        for t in range(update_interval): 
            print(f'Running game... Time step {self.env.time}\nState = {self.env.state}')

            # collect a set of trajectories D by running policy 
            action, prob = self.agent.policy(prev_state)

            print('action:',action, ' prob:', prob)

            expected_return = self.agent.value(prev_state)
            expected_return_list.append(expected_return)

            if (self.env.is_move_legal(action, prev_state)):
                self.env.update_env(action)
            trajectory.append((prev_state, action, prob))

            self.env.update_map(prev_state, self.env.state)
            self.env.render()

            # log the rewards  
            if (self.env.state['pos'] == self.env.goal):
                finish_reward = 10 

            reward_list.append(self.env.reward(prev_state, self.env.state) + finish_reward)
            
            prev_state = self.env.state
            self.env.time += 1

            if (self.env.reached_goal):
                break

        return trajectory, reward_list, expected_return_list


    def update_loop(self, num_trajectories):
        '''
        runs t times after each epoch (when the game ends)
        '''
        policy_gradient = 0
        trajectory_gradient = 0
        value_gradient = 0
        t_interval = 0
        num_trajectories -= 1

        for t in range(self.env.time):

            # trajectory_gradient = torch.autograd.grad(
            #     outputs=self.trajectories[t][2], # outputs are the probs of each action
            #     inputs=self.agent.policy_network.parameters(),  
            #     create_graph=False    
            # )
            #self.optimizer_1.zero_grad()
            #self.optimizer_2.zero_grad()

            print(self.trajectories[num_trajectories][t])

            self.trajectories[num_trajectories][t][2].backward()
            #trajectory_gradient = self.trajectories[num_trajectories][t][2].grad

            policy_params = nn.utils.parameters_to_vector(self.agent.policy_network.parameters())
            trajectory_gradient = nn.utils.parameters_to_vector(p.grad for p in self.agent.policy_network.parameters())

            print(trajectory_gradient.shape)

             # compute the gradient using torch
            advantage = self.calc_adv(self.calc_rewards_to_go(self.reward_lists[num_trajectories], t), self.agent.value(self.trajectories[num_trajectories][t][0]))
            policy_gradient += trajectory_gradient * advantage # element wise multi?

            print('adv shape: ', advantage.shape)
            print('policy gradient shape:', policy_gradient.shape)

            policy_params -= self.lr * policy_gradient
            nn.utils.vector_to_parameters(policy_params, self.agent.policy_network.parameters()) # load params back into model

            print(self.expected_return_lists[num_trajectories][t])
            print(self.reward_lists[num_trajectories][t])

            value_loss = self.calc_value_loss(self.expected_return_lists[num_trajectories][t], self.reward_lists[num_trajectories][t])
            value_loss.backward(retain_graph=True)


            self.optimizer_2.step()

            print('value loss:', value_loss)

        return policy_gradient, value_gradient

    def play_until_game_ends(self, update_interval):
        policy_gradient_k = 0
        value_gradient_k = 0
        num_trajectories = 0
        total_time = 0

        while not self.env.reached_goal:
            num_trajectories += 1
            self.env.time = 0

            trajectory, reward_list, expected_return_list = self.game_loop(update_interval=update_interval)
            self.trajectories.append(trajectory)
            self.reward_lists.append(reward_list)
            self.expected_return_lists.append(expected_return_list)

            policy_gradient, value_gradient, t_interval =  self.update_loop(num_trajectories)

            policy_gradient_k += policy_gradient
            value_gradient_k += value_gradient
            policy_gradient_k /= num_trajectories
            value_gradient_k /= (num_trajectories * self.env.time)

            with torch.no_grad():  # Disable gradient tracking for manual updates
                for param, grad in zip(self.agent.policy_network.parameters(), policy_gradient_k): # [issue] zip is too inefficient
                    param -= self.lr * grad  # Gradient descent update
            with torch.no_grad(): 
                for param, grad in zip(self.agent.value_network.parameters(), value_gradient_k):
                    param -= self.lr * grad  # Gradient descent update
            
            total_time += self.env.time

        self.game_summary()

    def game_summary(self):
        print('\nGame Complete =============================================')
        print(f'Total time steps {self.env.time}\nTotal rewards received{sum(self.reward_list)}\n')