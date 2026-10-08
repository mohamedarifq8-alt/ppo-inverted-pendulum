# 🚀 Mastering PPO: The Sanity Check (InvertedPendulum-v4)

This repository documents the foundational step in Deep Reinforcement Learning (RL): solving the classic Inverted Pendulum continuous control problem using the **Proximal Policy Optimization (PPO)** algorithm. Serving as the ultimate **"Sanity Check,"** this project ensures that the physics engine (MuJoCo), the cloud environment (Kaggle), and the RL pipeline are perfectly aligned before tackling highly complex robotic tasks.

---

## 🛤️ The Journey (Project Phases)

This project was executed systematically to establish a robust framework for continuous control tasks:

### Phase 1: Problem Formulation (MDP Analysis)
* **Description:** Philosophically and mathematically breaking down the environment before writing any code.
* **Techniques Used:** 
  * Analyzing the **Continuous Action Space** (applying horizontal force bounds `[-3.0, 3.0]` to a single joint).
  * Understanding the **4-dimensional Observation Space** (Cart Position, Pendulum Angle, Cart Velocity, Angular Velocity).
  * Mapping the **Reward Shaping** philosophy: *"Survival is winning"* (+1 reward for every step the pole remains upright, with implicit penalties for termination).
* **Result:** A clear architectural blueprint determining that a standard MLP policy is perfectly suited for this low-dimensional, fully observable state.

### Phase 2: Headless Cloud Training (Kaggle)
* **Description:** Setting up the infrastructure to train MuJoCo physics models on cloud servers without physical displays.
* **Techniques Used:**
  * Utilizing `Xvfb` and `pyvirtualdisplay` to create a virtual display for the OpenGL context required by MuJoCo.
  * Integrating **Stable Baselines3** standard PPO implementation with **Gymnasium**.
  * Using default, highly-optimized hyperparameters standard for benchmark environments.
* **Result:** Rapid convergence in just **25,000 timesteps**, reaching the maximum theoretical reward threshold.

### Phase 3: Deterministic Evaluation & Cinematic Rendering
* **Description:** Extracting the learned policy and rendering a visual proof of the agent's success.
* **Techniques Used:**
  * Setting `deterministic=True` during the prediction phase to strip away exploration noise, relying purely on the optimized policy.
  * Bypassing standard wrapper bugs (like `VecVideoRecorder` FPS issues) by manually extracting `rgb_array` frames via `env.render()`.
  * Compiling the frames into a high-quality video using `imageio` with `macro_block_size=None` to avoid FFMPEG dimensions errors.
* **Result:** Achieved **Mathematical Equilibrium**—the cart completely stopped moving while keeping the pendulum perfectly vertical for a full 1000-step episode (**Max Score: 1000 points**).

---

## 🧠 Core Concepts Explored

* **Continuous vs. Discrete Control:** Transitioning from button-mashing (Discrete) to applying precise, continuous joint torques.
* **Termination vs. Truncation:** Differentiating between a catastrophic failure (angle > 11 degrees) and a successful time-limit truncation (surviving 1000 steps) to ensure stable policy updates.
* **Deterministic Exploitation:** Understanding how exploration variance is crucial during training but detrimental during evaluation.
* **Headless Rendering Mechanics:** Mastering the virtual display integration to synchronize MuJoCo's physics clock with video compilation tools.

---

## 🛠️ Installation & Repository Structure

### Prerequisites

```bash
# Install system virtual display server for headless rendering
apt-get update && apt-get install -y xvfb

# Install Python packages supporting modern Gymnasium interface
pip install torch numpy gymnasium[mujoco] stable-baselines3[extra] imageio imageio-ffmpeg pyvirtualdisplay
```

### Repository Structure

```text
📦 PPO-InvertedPendulum
 ┣ 📜 InvertedPendulum.py              # Unified pipeline: Virtual Display, SB3 PPO Training & Cinematic Rendering
 ┣ 📜 ppo_inverted_pendulum.zip         # Saved Weights (Final SB3 Production Model)
 ┣ 📜 mujoco_inverted_pendulum.mp4      # Exported Video (1000-step perfect balance performance)
 ┗ 📜 README.md
```

---


