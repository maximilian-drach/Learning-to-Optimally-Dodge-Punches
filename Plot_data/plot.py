import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DDPG_normal_epi = pd.read_csv('DDPG_normal_epi.csv')
DDPG_ou_epi = pd.read_csv('DDPG_ou_epi.csv')
TD3_normal_epi = pd.read_csv('TD3_normal_epi.csv')
TD3_ou_epi = pd.read_csv('TD3_ou_epi.csv')
DDPG_normal_step = pd.read_csv('DDPG_normal_step.csv')
DDPG_ou_step = pd.read_csv('DDPG_ou_step.csv')
TD3_normal_step = pd.read_csv('TD3_normal_step.csv')
TD3_ou_step = pd.read_csv('TD3_ou_step.csv')


# fig, ax1 = plt.subplots(figsize=(10, 6))

# ax1.plot(DDPG_normal_epi['Step'], DDPG_normal_epi['Value'], linestyle='-', color='b', label='DDPG Normal (Episode)')
# ax1.plot(DDPG_ou_epi['Step'], DDPG_ou_epi['Value'], linestyle='-', color='r', label='DDPG OU (Episode)')
# ax1.plot(TD3_normal_epi['Step'], TD3_normal_epi['Value'], linestyle='-', color='g', label='TD3 Normal (Episode)')
# ax1.plot(TD3_ou_epi['Step'], TD3_ou_epi['Value'], linestyle='-', color='c', label='TD3 OU (Episode)')

# ax1.set_title('Average Episode Reward')
# ax1.set_xlabel('Step')
# ax1.set_ylabel('Value')
# ax1.legend()
# ax1.grid(True)


# fig, ax2 = plt.subplots(figsize=(10, 6))

# ax2.plot(DDPG_normal_step['Step'], DDPG_normal_step['Value'], linestyle='-', color='b', label='DDPG Normal (Step)')
# ax2.plot(DDPG_ou_step['Step'], DDPG_ou_step['Value'], linestyle='-', color='r', label='DDPG OU (Step)')
# ax2.plot(TD3_normal_step['Step'], TD3_normal_step['Value'], linestyle='-', color='g', label='TD3 Normal (Step)')
# ax2.plot(TD3_ou_step['Step'], TD3_ou_step['Value'], linestyle='-', color='c', label='TD3 OU (Step)')

# ax2.set_title('Average Step Reward')
# ax2.set_xlabel('Step')
# ax2.set_ylabel('Value')
# ax2.legend()
# ax2.grid(True)

# # Step 4: Show the plot
# plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_subplot(episode_data, step_data, title, noise_type):
    plt.subplot(1, 2, 1)
    plt.plot(episode_data['Step'], episode_data['Value'], linestyle='-', color='b', label=f'Average Episode Reward')
    plt.xlabel('Number of Steps')
    plt.ylabel('Average Reward Value')
    plt.legend(loc='lower left')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(step_data['Step'], step_data['Value'], linestyle='-', color='r', label=f'Average Step Reward (evaluated every 5000 steps)')
    plt.xlabel('Number of Steps')
    plt.ylabel('Average Reward Value')
    plt.legend(loc='lower left')
    plt.grid(True)

    plt.suptitle(f'{title} - {noise_type}')
    plt.tight_layout()

# Read data
DDPG_normal_epi = pd.read_csv('DDPG_normal_epi.csv')
DDPG_ou_epi = pd.read_csv('DDPG_ou_epi.csv')
TD3_normal_epi = pd.read_csv('TD3_normal_epi.csv')
TD3_ou_epi = pd.read_csv('TD3_ou_epi.csv')

DDPG_normal_step = pd.read_csv('DDPG_normal_step.csv')
DDPG_ou_step = pd.read_csv('DDPG_ou_step.csv')
TD3_normal_step = pd.read_csv('TD3_normal_step.csv')
TD3_ou_step = pd.read_csv('TD3_ou_step.csv')

# Figure 1: DDPG Normal
plt.figure(figsize=(12, 6))
create_subplot(DDPG_normal_epi, DDPG_normal_step, 'DDPG', 'Normal')
plt.show()

# Figure 2: DDPG OU
plt.figure(figsize=(12, 6))
create_subplot(DDPG_ou_epi, DDPG_ou_step, 'DDPG', 'OU')
plt.show()

# Figure 3: TD3 Normal
plt.figure(figsize=(12, 6))
create_subplot(TD3_normal_epi, TD3_normal_step, 'TD3', 'Normal')
plt.show()

# Figure 4: TD3 OU
plt.figure(figsize=(12, 6))
create_subplot(TD3_ou_epi, TD3_ou_step, 'TD3', 'OU')
plt.show()
