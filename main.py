import math 
from torch import nn
import torch
import numpy as np

'''
alt way to prevent agent from taking illegal movel: 
valid_mask = torch.tensor([1, 0, 1, 1])  # 1 = valid, 0 = invalid
masked_logits = logits + (1 - valid_mask) * -1e9  # Assign large negative value to invalid actions

# Sample action from masked probabilities
action_probs = torch.softmax(masked_logits, dim=0)
action = torch.multinomial(action_probs, num_samples=1)
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

        #self.head = nn.Softmax()

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32)
        logits = self.MLP(x)
        return logits

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

    def mask_illegal_actions(self, logits, state):
        possible_actions = [1,2,3,4]

        for i in range(len(logits)):
            for j in range(len(possible_actions)):
                if not self.is_action_legal(possible_actions[j], state):
                    logits[i][j] -= torch.inf

    def is_action_legal(self, action, state):
        
        if state['pos'][0] == 0 and action == 1:
            return False
        elif state['pos'][0] == 9 and action == 3:
            return False
        elif state['pos'][1] == 0 and action == 4:
            return False
        elif state['pos'][1] == 9 and action == 2:
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
        #print('\n')

class Game:
    def __init__(self):
        self.env = Environment()
        self.agent = Agent()
        self.trajectory_history = []
        self.reward_history = []
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
    
    def calc_value_loss(self, expected_return, reward): # use r to go instead

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
            #penalty = -2.5
            print(f'\n\nRunning game... Time step {self.env.time}\nState = {self.env.state}')

            #if self.env.time == 20:
            #    raise Exception('stopppppppppp')

            # collect a set of trajectories D by running policy 
            action, logits = self.agent.policy(prev_state)
            action = action.item()
            print('action:',action, ', prob:', nn.functional.softmax(logits.detach()))

            expected_return = self.agent.value(prev_state)
            expected_return_list.append(expected_return)

            print('expected_return:', expected_return)

            if self.env.is_move_legal(action, prev_state):
                self.env.update_env(action) # updates the state 
                #penalty = 0
                #print('updated state:', self.env.state)
            else:
                print('Agent moving into wall! AHHHHHHHHHHHHHHHHHHHH')
            
            self.env.mask_illegal_actions(logits, prev_state)
              
            trajectory.append((prev_state, action, logits))

            if (self.env.state['pos'] == self.env.goal):
                finish_reward = 100

            reward = self.env.reward(prev_state, self.env.state)
            print('reward:', reward)
            reward_list.append(reward + finish_reward)

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
        self.reward_history.append(reward_list)
        self.expected_return_history.append(expected_return_list)

        # some reshaping
        expected_return_list = torch.stack(expected_return_list)
        expected_return_list = expected_return_list.squeeze()

        advantage = self.calc_adv(self.calc_rewards_to_go(update_interval, reward_list), expected_return_list)

        print('advantage:', advantage)

        # print(expected_return_list)

        #print(advantage)
        action_logits = torch.stack([t[2] for t in trajectory])
        #print(action_probs)
        action_logits = torch.log_softmax(action_logits, dim=1)

        #print('advantage:', advantage)

        policy_loss_vector = -(advantage * action_probs)
        policy_loss_avg = torch.mean(policy_loss_vector)
        policy_loss_avg.backward()
        self.policy_optimizer.step()

        #print('policy loss shape:', policy_loss_vector.shape)
        print('policy loss avg:', policy_loss_avg)

        print(expected_return_list)
        print()
        print(reward_list)
        value_loss_vector = self.calc_value_loss(expected_return_list, reward_list)

        #print('value loss shape:', value_loss_vector.shape)

        value_loss_avg = torch.mean(value_loss_vector)


        print('value loss avg:', value_loss_avg)
        value_loss_avg.backward()
        self.value_optimizer.step()

        #  print(value_loss_avg)

    def play_n_games(self, update_interval, trajectory_limit, num_games):

        for n in range(num_games):
            print(f"\n\n{'=' * 50}\n|{'Running Game Number ' + str(n):^48}|\n{'=' * 50}")
            #while not self.env.reached_goal:
            for trajectory in range(trajectory_limit):
                
                self.trajectory_count += 1
                trajectory, reward_list, expected_return_list = self.game_loop(update_interval)
                self.update(update_interval, trajectory, reward_list, expected_return_list)

                if self.env.reached_goal:
                    #print('agent finished game!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    raise Exception('agent finished game!!!!!!!!!!!!!!!!!!!!!')
                    #break
                    
            
            if not self.env.reached_goal:
                self.env.x = 0
                self.env.y = 0
                self.env.state['pos'] = (self.env.x, self.env.y)
                self.env.update_map(self.env.state, self.env.state)

            self.game_summary(n)

    def game_summary(self, game_number):
        print(f"\n{'+' * 50}\n|{f'Game Number {game_number} Complete (Yay!)' + str():^48}|\n{'+' * 50}")
        print(f'Total time steps {self.env.time}\nTotal trajectories {self.trajectory_count}\nTotal rewards received: {sum(sum(i) for i in self.reward_history)}\n')