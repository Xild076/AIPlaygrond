import math
import sys
import time
import _pickle as CPickle
from tqdm import tqdm
import numpy as np
import datetime
import random
import pygame


class QBot(object):
    #Le brute force method of reinforcement learning (no me gusta pero es el sólo algoritmo que yo conozco)
    def __init__(self, observation_space, input_dim, learning_rate, discount_val, random_chance):
        self.observation_space = observation_space
        self.input_dim = input_dim
        self.q_table = np.zeros([self.observation_space, self.input_dim])
        self.learning_rate = learning_rate
        self.discout_val = discount_val
        self.random_chance = random_chance

    def choose_action_learn(self, state):
        #Take action with exploration
        if random.uniform(0, 1) < self.random_chance:
            action = random.randint(0, self.input_dim - 1)
        else:
            action = np.argmax(self.q_table[state])
        return action

    def choose_action_test(self, state):
        #Take action no exploration
        return np.argmax(self.q_table[state])

    def update_table(self, old_state, new_state, action, reward):
        #Update Q table
        old_value = self.q_table[old_state, action]
        next_max = np.max(self.q_table[new_state])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discout_val * next_max)
        self.q_table[old_state, action] = new_value

    def sl_model(self, sl_query):
        #Save load model, not working because mem too high (~ 21 GB)
        print(sys.getsizeof(self.q_table))
        if sl_query == 's':
            file = open('saved_model.txt', 'w')
            file.write(CPickle.dumps(self.q_table))
            file.close()
        if sl_query == 'l':
            file = open('saved_model.txt', 'r')
            item = file.read()
            if len(item) == 0:
                pass
            else:
                self.q_table = CPickle.loads(file.read())
            file.close()


class World(object):
    #Environment for ai

    class Player(object):
        #Interactable object
        def __init__(self, loc, direction, speed, turn_amount):
            self.action_space = [0, 1, 2]
            # 0 move forward, 1 turn left, 2 turn right
            self.loc = loc
            self.direction = direction
            self.speed = speed
            self.turn_amount = turn_amount

        def calc_new_loc(self):
            x_change = math.cos(self.direction) * self.speed
            y_change = math.sin(self.direction) * self.speed
            return [self.loc[0] + x_change, self.loc[1] + y_change]

        def take_action(self, action):
            if action == 0:
                self.loc = self.calc_new_loc()
            if action == 1:
                self.direction += self.turn_amount
            if action == 2:
                self.direction -= self.turn_amount
            if self.direction >= 360:
                self.direction = 0
            if self.direction < 0:
                self.direction = 360 - self.direction

    def __init__(self, size, scale):
        self.size = size
        self.scale = scale
        self.player = World.Player([random.random() * size * scale, random.random() * size * scale], 0, self.scale / 3, 10)
        rel_loc_ref = self.calc_rel_loc(self.player.loc)
        self.goal_loc = self.find_new_no_overlap([rel_loc_ref])

    def get_state_size(self):
        #with current parameters about 10^9
        scale_minloc = math.ceil(math.log10(self.size*self.scale))
        scale_maxloc = math.ceil(math.log10(self.size))
        print(10 ** (3 + scale_minloc*2 + scale_maxloc*2))
        return 10 ** (3 + scale_minloc*2 + scale_maxloc*2)

    def get_state(self):
        state = 0
        state += self.player.direction
        state += round(3 + self.player.loc[0])
        state += round(3 + self.player.loc[1]) * 10 ** (math.ceil(math.log10(self.size*self.scale)))
        state += self.goal_loc[0] * 10 ** (3 + math.ceil(math.log10(self.size*self.scale)) + math.ceil(math.log10(self.size)))
        state += self.goal_loc[1] * 10 ** (3 + math.ceil(math.log10(self.size*self.scale)) + 2 * math.ceil(math.log10(self.size)))
        return state

    def calc_rel_loc(self, loc):
        return [loc[0] // self.scale, loc[1] // self.scale]

    def find_new_no_overlap(self, list_overlap):
        obj_loc = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
        while obj_loc in list_overlap:
            obj_loc = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
        return obj_loc

    def calc_action_possible(self, action):
        possible = True
        if action == 0:
            new_loc = self.player.calc_new_loc()
            new_rel_loc = self.calc_rel_loc(new_loc)
            if 0 > new_rel_loc[0] or new_rel_loc[0] > self.size - 1:
                possible = False
            if 0 > new_rel_loc[1] or new_rel_loc[1] > self.size - 1:
                possible = False
        return possible

    def check_win(self):
        win = False
        rel_loc = self.calc_rel_loc(self.player.loc)
        if self.goal_loc == rel_loc:
            win = True
        return win

    def take_action(self, action):
        done = False
        reward = -5
        old_pos = self.player.loc
        if self.calc_action_possible(action):
            self.player.take_action(action)
        state = self.get_state()
        new_pos = self.player.loc
        if math.dist(old_pos, self.goal_loc) > math.dist(new_pos, self.goal_loc):
            reward = 10
        if self.check_win():
            done = True
            reward = 500
        return state, reward, done


SIZE = 4
SCALE = 4
EPOCHS = 10000
STEPS = 1000

world = World(SIZE, SCALE)
world.get_state_size()
bot = QBot(world.get_state_size(), len(world.player.action_space), 0.1, 0.8, 0.2)
wins = 0


for i in range(EPOCHS):
    done = False
    world.__init__(SIZE, SCALE)
    for _ in tqdm(range(STEPS), desc=f"Epoch: {i}"):
        before_state = world.get_state()
        action = bot.choose_action_learn(before_state)
        new_state, reward, done = world.take_action(action)
        bot.update_table(before_state, new_state, action, reward)
        if done:
            wins += 1
            break

pygame.init()

surface = pygame.display.set_mode((SCALE * SIZE * 25, SCALE * SIZE * 25))

red = (255, 0, 0)
green = (0, 255, 0)

for _ in range(10):
    done = False
    world.__init__(SIZE, SCALE)
    final_reward = 0
    final_steps = 0
    while not done and final_steps < 1000:
        surface.fill((255, 255, 255))
        pygame.draw.rect(surface, red,
                         pygame.Rect(world.goal_loc[0] * SCALE * 25, world.goal_loc[1] * SCALE * 25, SCALE * 25,
                                     SCALE * 25))
        pygame.draw.rect(surface, green,
                         pygame.Rect(world.player.loc[0] * 25, world.player.loc[1] * 25, SIZE * 2, SIZE * 2))
        pygame.display.flip()
        state = world.get_state()
        action = bot.choose_action_test(state)
        new_state, reward, done = world.take_action(action)
        bot.update_table(state, new_state, action, reward)
        final_reward += reward
        final_steps += 1
        time.sleep(.01)
        if done:
            print("WWWWWWWWWW")
            break

print(f"FINAL REWARD: {final_reward}, WINS: {wins}")

stats = open('stats.txt', 'a')
stats.write(f"Time: {datetime.datetime.now()} / Leaning Rate: {bot.learning_rate} / Discount Val: {bot.discout_val} / "
            f"Random Chance: {bot.random_chance} / Final Reward: {final_reward} \n")