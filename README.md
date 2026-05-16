
# Smart Taxi Autonomous Navigation using Q-Learning

An implementation of a self-driving taxi agent using **Q-Learning** (Reinforcement Learning) to navigate a gridworld environment, pick up a passenger, and drop them off at a designated location efficiently. 

This project uses OpenAI Gym's classic `Taxi-v3` environment to demonstrate how an agent transitions from complete random exploration to optimal, deterministic decision-making.

---

## 📌 Project Overview

The `Taxi-v3` environment consists of a $5 \times 5$ grid. There are 4 designated locations marked as **R**ed, **G**reen, **Y**ellow, and **B**lue. 

- **Goal:** The taxi must spawn at a random location, drive to the passenger, pick them up, drive to the destination, and drop them off.
- **State Space:** 500 total discrete states (combinations of taxi positions, passenger locations, and destination locations).
- **Action Space:** 6 discrete actions:
  - `0`: Move South (Down)
  - `1`: Move North (Up)
  - `2`: Move East (Right)
  - `3`: Move West (Left)
  - `4`: Pickup Passenger
  - `5`: Drop-off Passenger

---

## 🛠️ Dependencies & Installation

To run this repository locally, make sure you have Python 3.10+ installed along with the following libraries:

```bash
pip install numpy gym pygame stable-baselines3

```

> **Note:** This project utilizes `gym` along with `pygame` for rendering the graphical/text-based grid interface.

---

## 🚀 Performance Comparison

### 1. The Random Baseline Approach (No Learning)

When actions are chosen completely at random (`env.action_space.sample()`), the agent wanders aimlessly, racking up heavy negative penalties for improper pickups/drop-offs and unnecessary movements.

* **Steps taken:** ~1,425 steps
* **Total Reward accumulated:** -5,400

### 2. The Q-Learning Approach (Trained for 1,000 Episodes)

By introducing a Q-table ($500 \times 6$) and updating values using the Bellman Equation with a learning rate ($\alpha = 0.618$), the agent quickly converges to an optimal policy.

* **Steps taken:** Minimised to the optimal shortest path
* **Total Reward accumulated:** Consistent positive rewards (+6 to +12)

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left( R + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)$$

---

## 📈 Training Logs

The agent's progress over 1,000 training episodes demonstrates quick stabilization and high efficiency:

```text
Episode: 100  | Total reward: 12
Episode: 200  | Total reward: 10
Episode: 300  | Total reward: 11
Episode: 400  | Total reward: 11
Episode: 500  | Total reward: 8
Episode: 600  | Total reward: 7
Episode: 700  | Total reward: 9
Episode: 800  | Total reward: 8
Episode: 900  | Total reward: 12
Episode: 1000 | Total reward: 6

```

---

## 💻 Code Structure & Usage

### Core Implementation Snip

Below is the core loop used to train the agent via Q-Learning over 1,000 episodes:

```python
import gym
import numpy as np

env = gym.make("Taxi-v3")
n_states = env.observation_space.n
n_actions = env.action_space.n

# Initialize Q-Table
Q = np.zeros([n_states, n_actions])

# Hyperparameters
episodes = 1000
alpha = 0.618

for episode in range(1, episodes + 1):
    done = False
    state = env.reset()
    
    while not done:
        # Exploit learned values
        action = np.argmax(Q[state])
        new_state, reward, done, info = env.step(action)
        
        # Q-Table Update Rule
        Q[state, action] += alpha * (reward + np.max(Q[new_state]) - Q[state, action])
        state = new_state

```

### Visualizing the Trained Agent

Run the evaluation block in the script to watch the smart taxi pick up and drop off the passenger in real-time text visualization:

```text
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
  (Pickup)
...
  (Dropoff)
Success! Passenger reached destination safely.

```

---

## 📝 Key Insights

* **Barriers:** The `|` symbols in the environment display represent walls. The agent learns never to hit these walls as it optimizes rewards.
* **Reward System:** The agent receives a baseline penalty of `-1` for every step taken, a penalty of `-10` for illegal pickup/drop-off actions, and a major reward of `+20` for a successful drop-off.

```

```



