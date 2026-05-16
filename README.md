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



