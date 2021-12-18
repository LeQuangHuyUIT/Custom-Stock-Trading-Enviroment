import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df, window_size=5):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.window_size = window_size
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(5, window_size+2), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        self.window_size, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        self.window_size, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        self.window_size, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        self.window_size, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        self.window_size, 'Volume'].values / MAX_NUM_SHARES,
        ])
        obs = np.append(frame, np.array([
                    self.balance / MAX_ACCOUNT_BALANCE,
                    self.max_net_worth / MAX_ACCOUNT_BALANCE,
                    self.shares_held / MAX_NUM_SHARES,
                    self.cost_basis / MAX_SHARE_PRICE,
                    # self.total_shares_sold / MAX_NUM_SHARES,
                    self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
                ]).reshape(-1,1), axis=1)

        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])

        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price

            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1
        # print(f"current_step: {self.current_step}, len(df): {len(self.df)}")
        if self.current_step > len(self.df) - (self.window_size + 1):
            self.current_step = 0
            done = True
            # print("\t\t\tBREAK")
        else:
            done = self.net_worth <= 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
        

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'Open'].values) - 6)
        
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')

def Random_games(env, visualize, test_episodes = 50, comment=""):
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action = [np.random.randint(3, size=1)[0], np.random.rand()]
            # print(action)
            state, reward, done,_ = env.step(action)
            # print(f"state {state.shape}")
            # break
            if done:
                average_net_worth += env.net_worth
                # average_orders += env.episode_ordersprint("episode: {}, net_worth: {}, average_net_worth: {}".format(episode, env.net_worth, average_net_worth/(episode+1)))
                break
        # break

# df = pd.read_csv(f'excel_fpt (1).csv')
# df = df.drop(columns=['<Ticker>','<DTYYYYMMDD>'])
# df = df.set_axis(['Open','High','Low','Close','Volume'], axis=1, inplace=False)

# env = StockTradingEnv(df, window_size=30)
# Random_games(env, visualize=True, test_episodes = 1, comment="")
