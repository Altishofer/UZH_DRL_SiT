# Standard library imports
import os
from collections import deque

# Third-party imports
import numpy as np
import torch
from procgen import ProcgenEnv  # ProcGen environments for RL training

# Local imports
import data_augs  # Data augmentation functions
import data_augmentation_functions as custom_augs # Custom data augmentation functions
from baselines import logger  # Logging utilities for training metrics
from baselines.common.vec_env.vec_monitor import VecMonitor  # Environment monitoring
from baselines.common.vec_env.vec_normalize import VecNormalize  # Observation normalization
from baselines.common.vec_env.vec_remove_dict_obs import VecExtractDictObs  # Dictionary observation extraction
from test import evaluate  # Evaluation function for testing the model
from ucb_rl2_meta import algo, utils  # RL algorithms and utilities
from ucb_rl2_meta.algo.drac import DrAC  # DrAC algorithm implementation
from ucb_rl2_meta.arguments import parser  # Command line argument parser
from ucb_rl2_meta.envs import VecPyTorchProcgen  # PyTorch-compatible environment wrapper
from ucb_rl2_meta.model import Policy, Policy_Sit, AugCNN  # Neural network models
from ucb_rl2_meta.storage import RolloutStorage  # Storage for rollout data


# The following code correspond to the argumetns that are added to the parser in arguments.py
# Add device ID argument to specify which GPU to use
parser.add_argument(
    '--device_id',
    type=int,
    default=1,
    help='device')

# Add choice argument to select which SiT (Shifts in Time) model variant to use
parser.add_argument(
    '--choice',
    type=int,
    default=0,
    help='whihc sit model to use 0 SiT, 1 for SiTs'
)

# Add augmentation choice argument to select augmentation type
parser.add_argument(
    '--aug_choice',
    type=int,
    default=0,
    help='which aug-type selected')

# Add flag to enable/disable SiT model usage
parser.add_argument(
    '--use_sit',
    default=True,
    help='use Sit model')

# Add flag to enable/disable PPO algorithm usage
parser.add_argument(
    '--use_ppo',
    default=True,
    help='use PPo algo')



# Dictionary mapping augmentation names to their corresponding function implementations

#### Dictionary with the data augmentation functions that are already implemented ######
aug_to_func = {
    'crop': data_augs.Crop,           # Random cropping augmentation
    'random-conv': data_augs.RandomConv,  # Random convolution augmentation
    # 'grayscale': data_augs.Grayscale,  # Grayscale conversion (disabled)
    
    # Additional augmentations that are commented out:
    # 'rotate': data_augs.Rotate,      # Rotation augmentation
    # 'cutout': data_augs.Cutout,      # Cutout augmentation
    # 'cutout-color': data_augs.CutoutColor,  # Colored cutout augmentation
    'color-jitter': data_augs.ColorJitter,  # Color jittering augmentation
    'flip': data_augs.Flip,           # Horizontal/vertical flipping
}


###### Custom Augmentations ###### 
### Code added by CJ ### 
"""aug_to_func = {
    'crop': data_augs.Crop,
    'random-conv': data_augs.RandomConv,
    'color-jitter': data_augs.ColorJitter,
    'flip': data_augs.Flip,
}"""


def train(args):
    """
    Main training function for the reinforcement learning agent.
    
    Args:
        args: Parsed command line arguments containing all training parameters
    """
    # Configure CUDA settings and check availability
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    # Enable anomaly detection for debugging gradients
    torch.autograd.set_detect_anomaly(True)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Setup logging directory and clean up any existing logs
    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    # Configure device for training (GPU if available, otherwise CPU)
    # torch.set_num_threads(1)  # Commented out thread limitation
    #device = torch.device("cpu" if not torch.cuda.is_available() and args.cuda else "cuda:" + str(args.device_id))
    device = torch.device("cuda:" + str(args.device_id) if torch.cuda.is_available() else "cpu")
    print("-------  device: ", args.device_id, "------")
    #  = torch.device("cuda:" + str(args.device_id))

    # Create log file name with experiment details
    log_file = '-{}-{}-reproduce-s{}'.format(args.run_name, args.env_name, args.seed)

    #####  1 -- Create ProcGen environment with specified parameters #####
 
    venv = ProcgenEnv(num_envs=args.num_processes, env_name=args.env_name, \
                      num_levels=args.num_levels, start_level=args.start_level, \
                      distribution_mode=args.distribution_mode)
    
    ## Each environment generates visual observations (RGB images) with shape (3, 64, 64) and a discrete possible action space.
    """venv = ProcgenEnv(
        num_envs=args.num_processes,  #--> Specifies how many parallel copies of the environment are created simultaneously.
        # Example 8 parallel environments = 8 levels or 8 scenarios being running in parallel. 
        env_name=args.env_name, #--> Just the name of the game to be used.
        num_levels=args.num_levels, #--> Define how many different levels ProcGen generates for training.
        start_level=args.start_level,  #--> Sets the starting level index for level generation.
        distribution_mode=args.distribution_mode #--> Control the difficulty and visual variability of the environment.
    )
    """
    # The output of venv is a batch of RGB observations with shape (num_processes, 3, 64, 64)
    # After creating venv, wrappers are applied to adapt it for PyTorch and logging,
    # this is necesary because the original environment does not provide observations in a format that is directly compatible with PyTorch models.

    #### 2 -  Wrap the environment with various utility wrappers: #####
    
    # 1. Extract RGB observations from dictionary observations
    venv = VecExtractDictObs(venv, "rgb")
    
    # 2. Monitor episode statistics (rewards, lengths, etc.)
    # We need to keep the statistics of the last 100 episodes to compute mean/median rewards for logging.
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100) 
    
    
    # 3. Normalize observations (disabled for observations, enabled for rewards)
    venv = VecNormalize(venv=venv, ob=False)
    
    # 4. Convert to PyTorch-compatible environment
    envs = VecPyTorchProcgen(venv, device)

    # Get observation shape for neural network architecture
    obs_shape = envs.observation_space.shape 
    print("Observation shape:", obs_shape)  # There should be (3, 64, 64) for ProcGen RGB observations

    ##### 3 -  Initialize the policy network based on whether SiT (Shifts in Time) is used #### 

    """If the --use_sit flag is set, the code initializes a policy network that incorporates the SiT architecture. 
       else it initializes a standard policy network without SiT. The actor_critic variable holds the policy network instance."""
    
    if args.use_sit:
        # Use SiT-enhanced policy network
        actor_critic = Policy_Sit(
            obs_shape,   # --> Observation shape from the environment
            envs.action_space.n, # --> Number of discrete actions available in the environment
            device,             
            hidden_size=args.hidden_size,  # Hidden layer size for the network
            choice=args.choice,  # SiT variant selection (0 for SiT, 1 for SiTs)
            base_kwargs={'recurrent': False})  # , 'hidden_size': args.hidden_size})
        
        # Set logging directory based on SiT choice
        if args.choice == 0:
            args.log_dir = "logs"
        elif args.choice == 1:
            args.log_dir = "logs"

    else:
        # Use standard policy network without SiT
        actor_critic = Policy(
            obs_shape,
            envs.action_space.n,
            base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size})

    # 3.2 Move the policy network to the specified device (GPU/CPU)
    actor_critic.to(device)

    ### 4 - Initialize rollout storage for collecting experience data ###

    # This stores observations, actions, rewards, and other data for training
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size,
                              aug_type=args.aug_type, split_ratio=args.split_ratio)
    
    """rollouts = RolloutStorage(
        args.num_steps,  --> Number of steps to store in each rollout
        args.num_processes, --> Number of parallel environments
        envs.observation_space.shape, --> Shape of observations from the environment ((3, 64, 64) for RGB images)
        envs.action_space, --> discrete actions available in the environment
        actor_critic.recurrent_hidden_state_size, --> Size of recurrent hidden state (0 if not recurrent)
        aug_type=args.aug_type, --> Type of data augmentation to apply during training
        split_ratio=args.split_ratio --> Ratio for splitting augmented vs original data
    )"""

    # Calculate batch size for mini-batch training
    batch_size = int(args.num_processes * args.num_steps / args.num_mini_batch)

    ### 5 - Initialize the RL agent based on the specified algorithm ### 

    # Multiple algorithm options are available, each with different approaches to augmentation and learning

    """The following code block selects and initializes the appropriate RL algorithm based on command-line arguments.
       The available algorithms include UCB DrAC, Meta-learning DrAC, RL² DrAC, standard PPO, and the default DrAC.
       Each algorithm has its own way of handling data augmentations and learning strategies."""
    
    """UCB-DrAC (Upper Confidence Bound Data-regularized Actor-Critic) is an advanced RL algorithm that uses UCB to select 
    data augmentations during training.
    The following configurarion selects the identity augmentation , which means no modification to the input data are applied."""

    if args.use_ucb:
        # UCB (Upper Confidence Bound) DrAC algorithm
        # Uses UCB for selecting data augmentations during training
        aug_id = data_augs.Identity  # Identity augmentation (no modification)
        aug_list = [aug_to_func[t](batch_size=batch_size)
                    for t in list(aug_to_func.keys())]  # Create list of all augmentation functions
        print("Using UCB-DrAC with augmentations:", list(aug_to_func.keys()))

        agent = UCBDrAC(  # Note: UCBDrAC class needs to be imported
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            aug_list=aug_list,
            aug_id=aug_id,
            aug_coef=args.aug_coef,
            num_aug_types=len(list(aug_to_func.keys())),
            ucb_exploration_coef=args.ucb_exploration_coef,
            ucb_window_length=args.ucb_window_length)

    elif args.use_meta_learning:
        # Meta-learning DrAC algorithm
        # Uses meta-learning to adapt augmentation strategies during training
        aug_id = data_augs.Identity  # Identity augmentation
        aug_list = [aug_to_func[t](batch_size=batch_size) \
                    for t in list(aug_to_func.keys())]  # All available augmentations

        # Initialize augmentation CNN for meta-learning
        aug_model = AugCNN() # This is the network that will learn which augmentations to apply.
        aug_model.to(device)

        # Create meta-learning DrAC agent
        agent = algo.MetaDrAC(
            actor_critic,
            aug_model,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            meta_grad_clip=args.meta_grad_clip,
            meta_num_train_steps=args.meta_num_train_steps,
            meta_num_test_steps=args.meta_num_test_steps,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            aug_id=aug_id,
            aug_coef=args.aug_coef)

    elif args.use_rl2:
        # RL² (RL-squared) algorithm with DrAC
        # Uses a meta-learner to learn how to learn across different environments
        aug_id = data_augs.Identity  # Identity augmentation
        aug_list = [aug_to_func[t](batch_size=batch_size)
                    for t in list(aug_to_func.keys())]  # All available augmentations

        # Define observation space for RL² meta-learner
        # Includes action space size + 1 (for reward signal)
        rl2_obs_shape = [envs.action_space.n + 1]
        
        # Initialize RL² learner with recurrent architecture
        rl2_learner = Policy(
            rl2_obs_shape,
            len(list(aug_to_func.keys())),  # Output size = number of augmentation types
            base_kwargs={'recurrent': True, 'hidden_size': args.rl2_hidden_size})
        rl2_learner.to(device)

        # Create RL² DrAC agent
        agent = algo.RL2DrAC(
            actor_critic,
            rl2_learner,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            args.rl2_entropy_coef,
            lr=args.lr,
            eps=args.eps,
            rl2_lr=args.rl2_lr,
            rl2_eps=args.rl2_eps,
            max_grad_norm=args.max_grad_norm,
            aug_list=aug_list,
            aug_id=aug_id,
            aug_coef=args.aug_coef,
            num_aug_types=len(list(aug_to_func.keys())),
            recurrent_hidden_size=args.rl2_hidden_size,
            num_actions=envs.action_space.n,
            device=device)
    elif args.use_ppo:
        # Standard PPO (Proximal Policy Optimization) algorithm
        # A popular on-policy RL algorithm with clipped surrogate objective
        aug_id = data_augs.Identity  # Identity augmentation
        # aug_func = aug_to_func[args.aug_type](batch_size=batch_size)  # Single augmentation (commented)
        aug_list = [aug_to_func[t](batch_size=batch_size)
                    for t in list(aug_to_func.keys())]  # All available augmentations
        
        # Create PPO agent
        agent = algo.PPO(
            actor_critic,
            args.clip_param,           # PPO clipping parameter
            args.ppo_epoch,            # Number of PPO epochs per update
            args.num_mini_batch,       # Number of mini-batches
            args.value_loss_coef,      # Value function loss coefficient
            args.entropy_coef,         # Entropy regularization coefficient
            lr=args.lr,                # Learning rate
            eps=args.eps,              # Adam optimizer epsilon
            max_grad_norm=args.max_grad_norm,  # Gradient clipping norm
            aug_id=aug_id,
            aug_func=aug_list,
            aug_coef=args.aug_coef,    # Augmentation loss coefficient
            env_name=args.env_name)    # Environment name for logging
    else:
        # Default algorithm: DrAC (Data-regularized Actor-Critic)
        # Uses data augmentation as a regularization technique for improved generalization
        aug_id = data_augs.Identity  # Identity augmentation (no modification)
        # aug_func = aug_to_func[args.aug_type](batch_size=batch_size)  # Single augmentation (commented)
        aug_list = [aug_to_func[t](batch_size=batch_size)
                    for t in list(aug_to_func.keys())]  # All available augmentations
        
        # Create DrAC agent
        agent = DrAC(
            actor_critic,
            args.clip_param,           # PPO clipping parameter
            args.ppo_epoch,            # Number of PPO epochs per update
            args.num_mini_batch,       # Number of mini-batches
            args.value_loss_coef,      # Value function loss coefficient
            args.entropy_coef,         # Entropy regularization coefficient
            lr=args.lr,                # Learning rate
            eps=args.eps,              # Adam optimizer epsilon
            max_grad_norm=args.max_grad_norm,  # Gradient clipping norm
            aug_id=aug_id,
            aug_func=aug_list,
            aug_type=args.aug_choice,  # Specific augmentation type selection
            aug_coef=args.aug_coef,    # Augmentation loss coefficient
            env_name=args.env_name)    # Environment name for logging

    # Checkpoint handling for resuming training from saved models
    checkpoint_path = os.path.join(args.save_dir, "agent" + log_file + ".pt")
    
    if os.path.exists(checkpoint_path) and args.preempt:
        # Resume training from existing checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # Load model and optimizer states
        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Resume from the epoch after the saved one
        init_epoch = checkpoint['epoch'] + 1
        
        # Configure logger with epoch suffix to indicate resumed training
        logger.configure(dir=args.log_dir, format_strs=['csv', 'stdout'], 
                        log_suffix=log_file + "-e%s" % init_epoch)
    else:
        # Start training from scratch
        init_epoch = 0
        logger.configure(dir=args.log_dir, format_strs=['csv', 'stdout'], 
                        log_suffix=log_file)

    # Initialize environment and rollout storage
    obs = envs.reset()  # Get initial observations from all environments
    rollouts.obs[0].copy_(obs)  # Store initial observations in rollout buffer
    rollouts.to(device)  # Move rollout storage to GPU/CPU

    # Initialize episode rewards tracking (using deque for efficient rolling window)
    episode_rewards = deque(maxlen=10)
    
    # Calculate total number of updates for the training loop
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    # Main training loop - iterate through updates
    for j in range(init_epoch, num_updates):
        # Set model to training mode
        actor_critic.train()
        
        # Collect rollout data for one update cycle
        for step in range(args.num_steps):
            # Sample actions from the current policy
            with torch.no_grad():  # Disable gradient computation for inference
                # Apply identity augmentation to current observations
                obs_id = aug_id(rollouts.obs[step])
                
                # Get action, value estimate, and hidden states from policy
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    obs_id, 
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Execute actions in environment and observe results
            obs, reward, done, infos = envs.step(action)

            # Extract episode statistics from environment info
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # Create masks for episode termination and bad transitions
            # Masks: 0.0 if episode ended, 1.0 if continuing
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            
            # Bad masks: 0.0 for transitions that should be ignored (e.g., timeouts)
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            # Store the transition data in rollout buffer
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        # Compute value function estimate for the last observation (for GAE computation)
        with torch.no_grad():
            obs_id = aug_id(rollouts.obs[-1])  # Apply identity augmentation to final observation
            next_value = actor_critic.get_value(
                obs_id, 
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        # Compute returns and advantages using Generalized Advantage Estimation (GAE)
        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        # Update UCB values if using UCB algorithm and not on first iteration
        if args.use_ucb and j > 0:
            agent.update_ucb_values(rollouts)
            
        # Update the agent's policy and value networks
        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        
        # Clear rollout buffer for next iteration
        rollouts.after_update()

        # Logging and evaluation at specified intervals
        total_num_steps = (j + 1) * args.num_processes * args.num_steps
        
        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            # Calculate total environment steps taken so far
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            
            # Print training progress to console
            print("\nUpdate {}, step {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}"
                  .format(j, total_num_steps,
                          len(episode_rewards), np.mean(episode_rewards),
                          np.median(episode_rewards), dist_entropy, value_loss,
                          action_loss))

            # Log training metrics
            logger.logkv("train/nupdates", j)
            logger.logkv("train/total_num_steps", total_num_steps)

            # Log loss values
            logger.logkv("losses/dist_entropy", dist_entropy)
            logger.logkv("losses/value_loss", value_loss)
            logger.logkv("losses/action_loss", action_loss)

            # Log training reward statistics
            logger.logkv("train/mean_episode_reward", np.mean(episode_rewards))
            logger.logkv("train/median_episode_reward", np.median(episode_rewards))

            ### Evaluation on the Full Distribution of Levels ###
            # Evaluate the current policy on test environments
            eval_episode_rewards = evaluate(args, actor_critic, device, aug_id=aug_id)

            # Log evaluation metrics
            logger.logkv("test/mean_episode_reward", np.mean(eval_episode_rewards))
            logger.logkv("test/median_episode_reward", np.median(eval_episode_rewards))

            # Write all logged values to file
            logger.dumpkvs()

        # Model saving at specified intervals and at the end of training
        if (j > 0 and j % args.save_interval == 0
            or j == num_updates - 1) and args.save_dir != "":
            
            # Create save directory if it doesn't exist
            try:
                os.makedirs(args.save_dir)
            except OSError:
                pass  # Directory already exists

            # Save model checkpoint including:
            # - Current epoch/update number
            # - Model state (actor-critic network weights)
            # - Optimizer state (for proper resuming)
            torch.save({
                'epoch': j,
                'model_state_dict': agent.actor_critic.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, os.path.join(args.save_dir, "agent" + log_file + ".pt"))


# Main execution block
if __name__ == "__main__":
    # Parse command line arguments and start training
    args = parser.parse_args()
    train(args)
