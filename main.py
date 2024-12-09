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
        x = torch.as_tensor(x, dtype=torch.float32)
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
        x = torch.as_tensor(x, dtype=torch.float32)
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
        #print('model output probs', probs)

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
    2, 4: right and left

    pos y + action % 2
    '''
    def update_env(self, action):
        if action == 1:
            self.x -= 1
        elif action == 2:
            self.y += 1
        elif action == 3:
            self.x += 1
        elif action == 4:
            self.y -= 1
        
        self.state['pos'] = (self.x, self.y) # integers are immutable in py, so when you update x and y, the x and y in state doesn't update

        if self.state['pos'] == self.goal:
            self.state['has_reached_goal'] = True

    def update_map(self, prev_state, state):
        self.map[prev_state['pos'][0]][prev_state['pos'][1]] = 0 
        self.map[state['pos'][0]][state['pos'][1]] = 'A'


    def is_move_legal(self, action, state):

        #print(type(action))
        if state['pos'][0] == 0 and action == 1:
            return False
        elif state['pos'][0] == 9 and action == 3:
            #print('What?????????????????????????????????')
            return False
        elif state['pos'][1] == 0 and action == 4:
            return False
        elif state['pos'][1] == 9 and action == 2:
            #print('What?????????????????????????????????')
            return False
        else:
            return True 
        
    def reward(self, prev_state, state):
        '''
        computes reward at time step t
        '''
       # print('prev_state:', prev_state)
        #print('state:', state)

        init_dist = self.distance(prev_state["pos"][0], prev_state["pos"][1], self.goal[0], self.goal[1])
        next_dist = self.distance(state["pos"][0], state["pos"][1], self.goal[0], self.goal[1])

        #print('init_dist:', init_dist)
        #print('next_dist:', next_dist)

        return (init_dist - next_dist) * 10

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
        self.trajectory_history = []
        self.reward_histroy = []
        self.expected_return_history = []
        self.discount  = 1
        self.lr = 0.01
        self.policy_optimizer = torch.optim.SGD(self.agent.policy_network.parameters(), lr=self.lr)
        self.value_optimizer = torch.optim.SGD(self.agent.value_network.parameters(), lr=self.lr)
        self.trajectory_count = 0

    def calc_rewards_to_go(self, update_interval, reward_list): 
        rewards_to_go_list = []
        for t in range(update_interval):
            rewards_to_go = 0
            
            for t_prime in range(t, len(reward_list)):
                rewards_to_go += math.pow(self.discount, t_prime) * reward_list[t_prime]
            rewards_to_go_list.append(rewards_to_go)

        return rewards_to_go_list

    def calc_adv(self, rewards_to_go, expected_return_list):
        rewards_to_go = torch.as_tensor(rewards_to_go)
        expected_return_list = expected_return_list.detach() # remember to detach to aviod updating value network parameters

        #print('r to go shape:', rewards_to_go.shape)
        #print('expected return shape:', expected_return_list.shape)

        return rewards_to_go-expected_return_list
    
    def calc_value_loss(self, expected_return, reward):

        #print(expected_return)
        reward = torch.as_tensor(reward)
        
        MSE = torch.pow(expected_return - reward, 2)
        return MSE
       
    def game_loop(self, update_interval):
        prev_state = self.env.state.copy() # remember dicts are mutable; when you change one, the other changes 
        finish_reward = 0

        trajectory = []
        reward_list = []
        expected_return_list = []

        for t in range(update_interval): 
            penalty = -10
            print(f'Running game... Time step {self.env.time}\nState = {self.env.state}')

            if self.env.time == 10:
                raise Exception('stopppppppppp')

            # collect a set of trajectories D by running policy 
            action, prob = self.agent.policy(prev_state)
            action = action.item()
            print('action:',action, ' prob:', prob)

            expected_return = self.agent.value(prev_state)
            expected_return_list.append(expected_return)

            print(prev_state)

            if self.env.is_move_legal(action, prev_state):
                self.env.update_env(action) # updates the state 
                penalty = 0
                #print('updated state:', self.env.state)
            else:
                print('Agent moving into wall! AHHHHHHHHHHHHHHHHHHHH')
              
            trajectory.append((prev_state, action, prob))

            if (self.env.state['pos'] == self.env.goal):
                finish_reward = 100

            reward = self.env.reward(prev_state, self.env.state)
            print('reward:', reward)
            reward_list.append(reward + finish_reward + penalty)

            self.env.update_map(prev_state, self.env.state)
            self.env.render()

            prev_state = self.env.state.copy()
            self.env.time += 1
            if (self.env.reached_goal):
                break

        return trajectory, reward_list, expected_return_list#, action_prob_list

    def update(self, update_interval, trajectory, reward_list, expected_return_list):
        '''
        updates gradients per batch of update_interval size  
        '''
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        self.trajectory_history.append(trajectory)
        self.reward_histroy.append(reward_list)
        self.expected_return_history.append(expected_return_list)

        # some reshaping
        expected_return_list = torch.stack(expected_return_list)
        expected_return_list = expected_return_list.squeeze()

        advantage = self.calc_adv(self.calc_rewards_to_go(update_interval, reward_list), expected_return_list)

        # print(expected_return_list)

        #print(advantage)
        action_probs = torch.stack([t[2] for t in trajectory])
        #print(action_probs)

        #print('advantage:', advantage)

        policy_loss_vector = advantage * action_probs
        policy_loss_avg = torch.mean(policy_loss_vector)
        policy_loss_avg.backward()
        self.policy_optimizer.step()

        #print('policy loss shape:', policy_loss_vector.shape)
        print('policy loss avg:', policy_loss_avg)

        value_loss_vector = self.calc_value_loss(expected_return_list, reward_list)

        #print('value loss shape:', value_loss_vector.shape)

        value_loss_avg = torch.mean(value_loss_vector)


        print('value loss avg:', value_loss_avg)
        value_loss_avg.backward()
        self.value_optimizer.step()

        #  print(value_loss_avg)

    def play_n_games(self, update_interval, num_games):

        for n in range(num_games):
            print(f"\n\n{'=' * 50}\n|{'Running Game Number ' + str(n):^48}|\n{'=' * 50}")
            while not self.env.reached_goal:
                self.trajectory_count += 1
                trajectory, reward_list, expected_return_list = self.game_loop(update_interval)
                self.update(update_interval, trajectory, reward_list, expected_return_list)

            self.game_summary()

    def game_summary(self, update_interval, game_number):
        print(f"\n{'+' * 50}\n|{f'Game Number {game_number} Complete (Yay!)' + str():^48}|\n{'+' * 50}")
        print(f'Total time steps {self.env.time}\nTotal trajectories {self.trajectory_count}\nTotal rewards received{sum(self.reward_list)}\n')