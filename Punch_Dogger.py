import random
import sys
import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time  # Import the time module
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback

class PunchDodgerEnv(gym.Env):
    def __init__(self, render_mode="human"):
        super(PunchDodgerEnv, self).__init__()
        
        self.grid_size = 10
        #creates a vector for 5 x,y cordinates
        self.observation_space = spaces.Box(low=-self.grid_size, high=self.grid_size, shape=(10,), dtype=np.float64)
        #makes the action space .25 for the maximum distance the robot can travel
        self.action_space = spaces.Box(low=-.25, high=.25, shape=(2,), dtype=np.float64)  # 2D action space
        #radius of the human torso
        self.target_radius = 0.2
        #radius of boxing gloves
        self.punch_radius = .07


        self.initial_target_position = np.array([np.random.uniform(.9,1.1), np.random.uniform(.9,1.1)])
        self.target_position = self.initial_target_position.copy()
        self.prev_target_position = self.target_position.copy()
        self.initial_target_position = self.initial_target_position + np.array([np.random.normal(0,.02), np.abs(np.random.normal(0,.04))])

        #gets the velocity from "Design and evaluation of a biomechanically consistentmethod for markerless kinematic analysis of sportsmotion"
        self.velocity = 11.3
        self.acceleration = -24

        self.punch_position = np.array([np.random.rand()*2, 0])
        self.prev_punch_position = self.punch_position.copy()

        #sets the frame rate of the camera
        self.framerate = 120
        self.timestep = 0

        #sees if the espisode is done variable
        self.done = False
        self.distance_to_target = np.linalg.norm(self.target_position - self.punch_position) - self.punch_radius - self.target_radius
        self.render_mode = render_mode

        self.i = 0
        #average jab .7 m
        #velocity = 11.3 m/s
        #velocyty_final = 8.9
        # time = 60-100 ms == .06 ~.1 sec
        #0.01666666666
        #acc = 14.3/ .1 = 143
        #60 fps == 1/60
        # 1/10 / 10/60  --> 6 frames
        # 1/10 / 1/100 --> 10 frames

    def reset(self, seed=0):
        np.random.seed(seed)
        #create a stochastic place where the robot spawns
        initial_target_pos = np.array([np.random.uniform(.5,1.5), np.random.uniform(.6,1.1)])

        #set the intial target location
        self.initial_target_position = initial_target_pos
        self.target_position = self.initial_target_position.copy()
        self.prev_target_position = self.target_position.copy()
        #creates stochasticisty in intial location, to more aligned with human target tracking
        self.initial_target_position = self.initial_target_position + np.array([np.random.normal(0,.02), np.abs(np.random.normal(0,.04))])
        #spwans the punch on the x-axis, but uniformly along its axis
        self.punch_position = np.array([np.random.uniform(0,2), 0])
        self.prev_punch_position = self.punch_position.copy()
        self.distance_to_target = np.linalg.norm(self.target_position - self.punch_position) - self.punch_radius - self.target_radius
        
        #gets the velocity from "Design and evaluation of a biomechanically consistentmethod for markerless kinematic analysis of sportsmotion"
        self.velocity = 11.3
        self.timestep = 0
        #average jab .7 m
        #velocity = 11.3 m/s
        #velocyty_final = 8.9

        self.done = False
        #returns the state vector
        return self._get_observation(), {}

    def step(self, action):
      
        #copies the previes target and punch location
        self.prev_target_position = self.target_position.copy()
        self.prev_punch_position = self.punch_position.copy()
        
        #sets the step reward to 0
        reward = 0

        #gets the new position of the target
        new_pos = self.target_position + action

        #Boundry Rewards
        if (new_pos[0] > 1.99) or (new_pos[0] < .01):
            reward -= 150 

        if (new_pos[1] > 1.99) or (new_pos[1] < .01):
            reward -= 150
        
        #copies the new position to the target location
        self.target_position = new_pos

        #gets the unit norm direction of the intial target with respect to the punch
        direction = self.initial_target_position - self.punch_position
        direction = direction / np.linalg.norm(direction)
        
        #calculates the velocity
        self.velocity += self.acceleration * self.timestep
        #increase the timestep
        self.timestep += 1/self.framerate
        #new punch location
        self.punch_position = self.prev_punch_position + direction * self.velocity * self.timestep
       

        #gets the unit norm direction of the target to the punch
        target_2_punch_dir = self.punch_position - self.target_position
        target_2_punch_dir =  target_2_punch_dir / np.linalg.norm(target_2_punch_dir)

        #gets the unit norm direction of the pucnh to the target
        punch_2_target_dir = self.target_position - self.punch_position
        punch_2_target_dir = punch_2_target_dir / np.linalg.norm(punch_2_target_dir)
       
        #calculates the distances between the closest points on the circumference of the punch and target
        self.distance_to_target = np.linalg.norm(self.target_position - self.punch_position) - self.punch_radius - self.target_radius
        distance_to_target = self.distance_to_target
        
        #ideal distance from the punch
        ideal_distance = .4
        ideal_tolerance = .1
        
        #Distance Reward
        #checkes if the distance between the closest points on the circumference of the punch and target is with the ideal distance range
        if((distance_to_target >= ideal_distance - ideal_tolerance) and (distance_to_target <= ideal_distance + ideal_tolerance)):
            #positive reward if it is
            reward += 10

        #negative reward if its too far from the punch
        elif(distance_to_target > ideal_distance + ideal_tolerance):
            reward += -10*(distance_to_target - ideal_distance)
        #negative reward if its too close to the punch
        elif(distance_to_target >= 0):
            reward += -10 * np.clip((1/abs(distance_to_target)), 0, 5) 
        #Collision Reward
        else:
            reward += -20 * np.clip((1/abs(distance_to_target)), 0, 5)

        prev_distance_to_target = np.linalg.norm(self.target_position - self.prev_punch_position) - self.punch_radius - self.target_radius
        #automatic -100 for bieng in punch trajectory
        #collision reward
        if(prev_distance_to_target <= .05):
            reward += -100
        
        #movement reward
        reward += -30*(abs(action[1])) + -1*(abs(action[0]))
        #checks to see if the punch is above the intial target positions y-cordinate, signifying that its punch trajectory has gone past its target
        if (self.initial_target_position[1] - self.punch_position[1] - self.punch_radius*direction[1])  < .01:
            self.done = True

        #returns the state space
        obs = np.concatenate([self.target_position, self.prev_target_position, self.punch_position, self.prev_punch_position, target_2_punch_dir])
        return obs, reward, self.done, False, {}

    def render(self, mode="human"):

        if mode=='human':
            # Visualize the environment
            plt.figure(figsize=(5, 5))
            plt.xlim(0, 2)
            plt.ylim(0, 2)

            # Plot the target state as a circle
            target_circle = Circle((self.target_position[0], self.target_position[1]), self.target_radius, color='g', fill=False)
            plt.gca().add_patch(target_circle)
            

            # Plot the previous target state location
            plt.plot(self.prev_target_position[0], self.prev_target_position[1], 'co', label='Prev Target Pos')
            plt.plot(self.target_position[0], self.target_position[1], 'go', label='Curr Target Pos')
            plt.plot(self.initial_target_position[0], self.initial_target_position[1], 'bx', label='Init Target Pos')
            plt.plot(self.punch_position[0], self.punch_position[1], 'ro', label='Punch Pos')
            plt.plot(self.prev_punch_position[0], self.prev_punch_position[1], 'mo', label='Prev Punch Pos')

            # Plot the punch state as a circle
            punch_circle = Circle((self.punch_position[0], self.punch_position[1]), self.punch_radius, color='r', fill=False)
            plt.gca().add_patch(punch_circle)

            plt.title(f'Punch Dodger Environment @ timestep {round(float(self.timestep),3)} sec')
            plt.xlabel('X-coordinate')
            plt.ylabel('Y-coordinate')
            plt.legend()
            # plt.show()

        elif mode=='rgb_array':
            #fluff to make sure I could run the enviorment
            raise NotImplementedError("Rendering mode 'rgb_array' not implemented yet")

        else:
            #fluff to make sure I could run the enviorment
            print("**")
            super().render(mode=mode)

    def _get_observation(self):
        # Return the current observation (concatenation of target and punch positions
        target_2_punch_dir = self.punch_position - self.target_position
        target_2_punch_dir =  target_2_punch_dir / np.linalg.norm(target_2_punch_dir)
        #return ths space state vector
        obs = np.concatenate([self.target_position, self.prev_target_position, self.punch_position, self.prev_punch_position, target_2_punch_dir])
        return obs
        

def create_directories(algorithm = "td3", noise='ou'):
    # Create "logs" directory if it doesn't exist
    if not os.path.exists(f"logs_{algorithm}_{noise}"):
        os.makedirs(f"logs_{algorithm}_{noise}")
        print("Created 'logs' directory.")

    # Create "TFlog" directory if it doesn't exist
    if not os.path.exists(f"TFlog_{algorithm}_{noise}"):
        os.makedirs(f"TFlog_{algorithm}_{noise}")
        print(f"Created 'TFlog_{algorithm}_{noise}' directory.")

def main():

    if len(sys.argv) != 4:
        #checks the input arguments
        print("Usage: python algorithm_selector.py <algorithm> <Normal/OU> <train/test/demo>")
        sys.exit(1)
    else:
        #checks that
        algorithm = sys.argv[1].lower()
        noise = sys.argv[2].lower()
        run = sys.argv[3].lower()
        
        if algorithm != 'ddpg' and algorithm != 'td3':
            print("Please select an algorithm ddpg or td3\nUsage: python algorithm_selector.py <ddpg/td3> <Normal/OU> <train/test/demo>")
            sys.exit(1)
        if noise != 'normal' and noise != 'ou':
            print("Please select an noise normal or ou\nUsage: python algorithm_selector.py <ddpg/td3> <Normal/OU> <train/test/demo>")
            sys.exit(1)
        if run != 'train' and run != 'test' and run != 'demo':
            print("Please select an action train or test or demo\nUsage: python algorithm_selector.py <ddpg/td3> <Normal/OU> <train/test/demo>")
            sys.exit(1)
        
        env = PunchDodgerEnv()
        check_env(env)
        env.reset()

        #will save a demo to a folder
        if run == "demo":
            if not os.path.exists(f"logs_{algorithm}_{noise}"):
                print(f"No model folder called logs_{algorithm}_{noise}")
                sys.exit(1)
            
            if algorithm == 'ddpg':
                #loads the pretrained model
                model = DDPG.load(f"./logs_{algorithm}_{noise}/best_model0/best_model.zip", env=env)
                vec_env = model.get_env()
            else:
                #loads the pretrained model
                model = TD3.load(f"./logs_{algorithm}_{noise}/best_model0/best_model.zip", env=env)
                vec_env = model.get_env()

            #makes an output demo folder
            output_folder = f"{algorithm}_{noise}_demo"
            os.makedirs(output_folder, exist_ok=True)
            simulate_again = 1
            episode_i = 0
            while simulate_again > 0:
                obs = vec_env.reset()
                dones = False
                step_i = 0
                #renders a figure
                vec_env.render()
                #saves the render figure to an image
                plt.savefig(os.path.join(output_folder, f"demo{episode_i}_{step_i}.png"))
                plt.close()
                while dones==False:
                    #gets the policy network output from inputted state space
                    action, _states = model.predict(obs)
                    #runs a steps with the inputted action vector
                    obs, rewards, dones, info = vec_env.step(action)
                    
                    #test to see if the run is done
                    if(dones == True):
                        break
                    #renders the action
                    vec_env.render()
                    step_i+=1
                    #saves the render to a image
                    plt.savefig(os.path.join(output_folder, f"demo{episode_i}_{step_i}.png"))
                    plt.close()
                simulate_again = float(input("Enter 1 to simulate, -1 to stop simulating\nDo wanna simulate again: "))
                #goes to a new episode
                episode_i += 1

        #will run the test plots
        elif run == 'test':

            if not os.path.exists(f"logs_{algorithm}_{noise}"):
                print(f"No model folder called logs_{algorithm}_{noise}")
                sys.exit(1)
            
            if algorithm == 'ddpg':
                model = DDPG.load(f"./logs_{algorithm}_{noise}/best_model0/best_model.zip", env=env)
                vec_env = model.get_env()
            else:
                model = TD3.load(f"./logs_{algorithm}_{noise}/best_model0/best_model.zip", env=env)
                vec_env = model.get_env()
            
            simulate_again = 0
            #stores the episode run results
            episode_reward_list = []
            average_step_reward = []

            while simulate_again < 1000:
                episode_reward = 0
                num_steps = 0
                #resets the enviorment
                obs = vec_env.reset()
                dones = False
                while dones==False:
                    #gets the output from the policy network
                    action, _states = model.predict(obs)
                    #runs the action vector in the enviorment
                    obs, rewards, dones, info = vec_env.step(action)
                    #stores the rewards from the step
                    episode_reward += rewards
                    num_steps += 1
                    if(dones == True):
                        break
                episode_reward_list.append(episode_reward)
                average_step_reward.append(episode_reward/num_steps)
                
                simulate_again += 1
            #plots the simulations statistics
            plt.plot(range(len(episode_reward_list)), episode_reward_list, color='r', label=f'Total Episode Reward')
            plt.plot(range(len(average_step_reward)), average_step_reward, color='g', label=f'Avg Step Reward per Episode')
            plt.xlabel("Episode") 
            plt.ylabel("Reward") 
            plt.title(f"{algorithm.upper()} w/ {noise.upper()} noise Episode vs Reward") 
            plt.text(1, 0.01, f"Episode (mean, std) {round(float(np.mean(episode_reward_list)),1)},{round(float(np.std(episode_reward_list)),1)}\nStep (mean, std) {round(float(np.mean(average_step_reward)),1)},{round(float(np.std(average_step_reward)),1)}", horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)
            plt.legend(loc='lower left')
            plt.show()

        else:
            #this Trains the model
            n_actions = env.action_space.shape[-1]
            if noise == 'normal':
                #gets the normal noise class
                action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1*np.ones(n_actions))
            else:
                #gets the OU noise class
                action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=.2*np.ones(n_actions))
            #creates the directories to store the models and tensorflow data
            create_directories(algorithm,noise)
            
            log_folder = f"./TFlog_{algorithm}_{noise}/"

            if algorithm == 'ddpg':
                #sets the model to use to be DDPG
                model = DDPG("MlpPolicy", env, learning_starts=300,action_noise=action_noise, tensorboard_log = log_folder, verbose=2)
            else:
                #sets the model to use to be TD3
                model = TD3("MlpPolicy", env, learning_starts=300,action_noise=action_noise, tensorboard_log = log_folder, verbose=2)
           
            #stops the training, if after 100 evaluations there is no improvement in the mean reward score
            stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=3, verbose=1)
            total_timesteps = 10000000

            #optimization option
            optimization = 0
            while optimization < 3:
                #create a callback model every 5000 steps and keeps the model with the best average reward score
                eval_callback = EvalCallback(env, best_model_save_path=f"./logs_{algorithm}_{noise}/best_model{optimization}/", log_path=f"./logs_{algorithm}_{noise}/results{optimization}/",eval_freq=5000, callback_after_eval=stop_train_callback, verbose=1)
                #calls a function that runs the model over the step to learn
                #will stop learning, if the there has been no imporvempent for the past 100 evaluations
                model.learn(total_timesteps=total_timesteps, log_interval=10, progress_bar=True, callback=eval_callback)
                # loads the best model from the training
                model = TD3.load(f"./logs_{algorithm}_{noise}/best_model{optimization}/best_model.zip", env=env)

                vec_env = model.get_env()
                simulate_again = 1

                while simulate_again > 0:
                    obs = vec_env.reset()
                    dones = False
                    vec_env.render()
                    plt.show()
                    while dones==False:
                        #gets the policy network action vector output
                        action, _states = model.predict(obs)
                        #inputs the action vector in the enviorment
                        obs, rewards, dones, info = vec_env.step(action)
                        
                        if(dones == True):
                            break
                        vec_env.render()
                        plt.show()
                    #sees if you want another simulation
                    simulate_again = float(input("Enter 1 to simulate, -1 to stop simulating\nDo wanna simulate again: "))
                #after seeing some simulation, ask if you want to reoptimize
                optimize_again = str(input("Enter y to reoptimize, n to stop and keep the model\nDo wanna optimize again: "))
                
                if optimize_again != 'y':
                    break
                else:
                    optimization += 1


if __name__ == "__main__":
    main()