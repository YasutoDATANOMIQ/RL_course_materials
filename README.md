# Main components of RL course 

![Alt text](images/RL_curriculum.png?raw=true "Course schedule")

Sample lecture slide
   - Lecture 1: https://www.slideshare.net/slideshows/reinforcement-course-material-samples-lecture-1/266653268
   - Lecture 2: https://www.slideshare.net/slideshows/how-to-formulate-reinforcement-learning-in-illustrative-ways/266653309


1. "RL" without "trial and errors"
   - Implementing Bellman-equation-like recurrent formula 
   - Policy iteration and value iteration on state transition diagrams or grid maps
2. Introducing "experiences" and "trial and errors" in RL
   - Policy evaluation by trial and errors in a grid map
   - Control with Monte-Carlo method (batch data of experiences)
   - Control with TD loss (online data of experiences, Q-learning, Sarsa)
3. Elaborating RL training
   - n-step TD, (optional) eligibility traces 
   - Exploration-exploitation tradeoff (first with an example of multi-armed bandit, and then grid map)
   - On-policy vs off-policy (for now only with an example of Q-learning, Sarsa comparison)
   - Model-free vs model-based (only Dyna as an example control, and other scripts are demos of searching)
   - Actor-critic structures (with an example with advantage function on grid map)
4. Making environments and agents richer
   - Comparisons of Action-value function approximation by tabular, linear functions, and neural networks
   - Stabilizing training of neural networks with experience replay
   - Policy gradient demo for episodic and continuous cases
5. Advanced and practical topics, implementations
   - A practical demo for RL in finance (ideally without deep learning)
   - A practical demo for video Game RL (ideally with Deep Q-Network with some techniques in Rainbow)
   - A practical demo with policy gradient and some RL libraries (e.g. tf agent)

