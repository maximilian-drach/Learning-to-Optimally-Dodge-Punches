# Learning to Optimally Dodge Punches: A Deep Reinforcement Learning Approach to Real-Time Robot Defense Strategies

Enviorment in is created in gymnasium
RL Algorithms are implimented using the stable_baseline3[extra] library 
Make sure to have tensorflow, gymnasium, sable_baseline3[extra], numpy, & matplotlib installed

### Usage
From the commandline you can choose betwen the:
Algorithm: DDPG / TD3
Noise: Normal / OU
Action: Train / Test / Demo
Train -> will train your model
Test -> will run 1000 episode test on your model
Demo -> will create a folder {algorithm}_{noise}_demo that will the outputted visually images from the algorithm
**If your going to test the model, make sure to have your pretrained models installed**
```
python3 Punch_Dogger.py ddpg ou train
```

You can also monitor the models performance while they train with the tensorflow board

```
tensorboard --logdir TFlog_{algorithm}_{noise}
```
Open the link in command terminal "http://localhost:xxxx/" to see the live meterics

