# Main components of RL course 

![Alt text](images/RL_curriculum.png?raw=true "Course schedule")


1. "RL" without "trial and errors"
   - Implementing Bellman-equation-like recurrent formula 
   - Policy iteration and value iteration on state transition diagmras or grid maps
2. Introducing "experiences" and "trial and errors" in RL
   - Policy evaluation by trial and errors in a grid map
   - Control with Monte-Carlo method (batch data of experiences)
   - Control with TD loss (online data of experiences, Q-learning, Sarsa)
3. Elaborating RL traning
   - n-step TD, (optional) eligibility traces 
   - Exploration-exploitation tradeoff (first with an example of multi-armed bandin, and then grid map)
   - On-policy vs off-policy (for now only with an example of Q-learning, Sarsa comparison)
   - Model-free vs model-based (only Dyna as an example control, and other scripts are demos of searching)
   - Actor-critic structures (with an example with advantage function on grid map)
4. Making environments and agents richer
   - Comparisons of Action-value function approximation by tabular, linear functions, and neural networks
   - Stabilizing training of neural networks with experience replay
   - Policy gradient demo for episodic and continuous cases
5. Advanced and practical topics, implementations
   - A practical demo for RL in finance (ideally without deep learning)
   - A practical demo for video Game RL (ideall with Deep Q-Netowork with some techniques in Rainbow)
   - A practical demo with policy gradient and some RL libraries (e.g. tf angent)

# Tasks left on each directory 

Note: directories are divided based on topics, not lecture parts

### General tasks
 - [ ] Cleaning and adjusting scripts so that they can be used for lectures
 - [ ] Checking copy rights and clarifying sources

### basic_MDP
 - [x] Preparing basic grid map environment and agent
 - [x] Changing data structure of states in basic MDP environemnts (Don't use defaultdict)
 - [ ] (Optional) To make a simple state-transit diagram MDP with Networkx
 - [ ] Introducing Open AI gym as an environment


### dynamic_programming
 - [x] Visualizing DP with grid map
 - [ ] (Optional) Making DP demo with state-transit diagrams
 - [ ] Cleaning up and commenting on code

### model_free_RL 
 - [x] Making Q-learning, SARSA 
 - [x] Animating the demo
 - [ ] (Optional) Visualizing action values as heatmaps
 - [ ] Making policy evaluation demo with a fixed policy
 - [ ] Making n-step TD demo 
 - [ ] (Optional) to make eligibility traces demo
 - [ ] Tabular actor-critic RL with advantage function
 - [ ] Cleaning up and commenting on code


### exploration_exploitation
 - [x] Exploitation vs exploration demo with multi-armed bandit
 - [x] UCB1 demo with multi-armed bandit
 - [x] (Optional) Finding double Q-learning demo
 - [ ] Exploitation vs exploration demo with Q-learning
 - [ ] Cleaning up and commenting on code

### model_based
 - [x] Fixing Dyna script
 - [ ] (Optional) Visualizing tree search with state transition diagram
 - [ ] Visualizing tree search with tic tac toe
 - [ ] Cleaning up and commenting on code

### function_approximation_value
 - [x] Finding implementation of control with linear function approximation
 - [ ] Stabilizing traiing of neural networks with experience replay
 - [ ] Cleaning up and commenting on code

### function_approximation_policy
 - [x] Finding implementation of policy gradient with neural nets
 - [ ] Supress warnings in baby_step scripts
 - [ ] Raising TF1 to TF2 in baby_step scripts
 - [ ] Finding implementation of policy gradient e.g. REINFORCE without neural nets
 - [ ] Cleaning up and commenting on code

### Advanced and practical RL
 - [ ] (Benjamin) Financial RL (ideally without neural networks)
 - [ ] (Benjamin) Video game RL (probably with Deep Q-Netowrk)
 - [ ] (Optional) Industrial RL with kuka simulator (hopefully with policy gradient and with TF Agent)
 - [ ] (Optional) Finding demos only to show for advanced topic such e.g. distributional neural nets
 - [ ] Cleaning up and commenting on code