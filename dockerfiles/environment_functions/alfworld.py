ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

def alfworld_rollout_once(
    trainer,
    env_id: str,
    env_endpoint: str,
    game_id: int,
    max_turns: int = 30,
    history_length: int = 2,
    timeout: int = 2400
) -> dict[str, list]:
    """Execute a single episode rollout in AlfWorld environment."""
    from trl.experimental.openenv import generate_rollout_completions
    import requests
    
    tokenizer = trainer.processing_class
    
    episode_prompt_ids: list[int] = []
    episode_completion_ids: list[int] = []
    episode_logprobs: list[float] = []
    invalid_count = 0
    done = False
    solved = False
    turn_number = 0
    
    # Memory to track history
    memory = []  # List of {"observation": ..., "action": ...}
    task_description = ""
    
    # --- Reset Environment (POST /reset) ---
    payload = {"id": env_id, "game": game_id, "world_type": "Text"}
    
    try:
        reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=timeout)
        reset_res.raise_for_status()
        reset_data = reset_res.json()
        
        # Extract task description from initial observation
        current_observation = reset_data["observation"]
        current_available_actions = reset_data["available_actions"]
        
        # Extract task
        task_start = current_observation.find('Your task is to: ')
        if task_start != -1:
            task_description = current_observation[task_start + len('Your task is to: '):].strip()
        else:
            task_description = "complete the task"
        
    except Exception as e:
        print(f"Failed to reset environment (Game {game_id}): {e}")
        return {
            "prompt_ids": [],
            "completion_ids": [],
            "logprobs": [],
            "env_reward": 0.0
        }

    # --- Interaction Loop ---
    while not done and (turn_number < max_turns):
        # Build formatted observation using templates
        if turn_number == 0 or history_length <= 0:
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in current_available_actions if s != 'help')
            formatted_observation = ALFWORLD_TEMPLATE_NO_HIS.format(
                current_observation=current_observation,
                admissible_actions=reformatted_admissible_actions
            )
        else:
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in current_available_actions if s != 'help')
            
            # Build action history from memory
            recent = memory[-history_length:] if len(memory) > history_length else memory
            valid_len = len(recent)
            start_idx = len(memory) - valid_len
            
            lines = []
            for j, rec in enumerate(recent):
                step_num = start_idx + j + 1
                obs = rec['observation']
                act = rec['action']
                lines.append(
                    f"[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
                )
            
            action_history = "\n".join(lines)
            
            formatted_observation = ALFWORLD_TEMPLATE.format(
                task_description=task_description,
                step_count=len(memory),
                history_length=valid_len,
                action_history=action_history,
                current_step=len(memory) + 1,
                current_observation=current_observation,
                admissible_actions=reformatted_admissible_actions
            )
        
        # Build messages for generation
        messages = [{"role": "user", "content": formatted_observation}]
        prompt_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Generate Rollout Completion
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids = rollout_outputs.get("prompt_ids", [])
        completion_ids = rollout_outputs.get("completion_ids", [])
        logprobs = rollout_outputs.get("logprobs", [])
        completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

        # Accumulate tokens across all turns
        episode_prompt_ids.extend(prompt_ids)
        episode_completion_ids.extend(completion_ids)
        episode_logprobs.extend(logprobs)

        # --- Parse Action ---
        action_to_send = completion_text
        if action_to_send.endswith("</s>"):
            action_to_send = action_to_send[:-5]

        # Parse action from <action> tags
        if "<action>" in action_to_send.lower() and "</action>" in action_to_send.lower():
            action_start = action_to_send.lower().find("<action>") + len("<action>")
            action_end = action_to_send.lower().find("</action>")
            action_to_send = action_to_send[action_start:action_end].strip()
        
        # --- Step Environment (POST /step) ---
        step_reward = 0.0
        step_done = False
        step_state = ""
        pre_observation = current_observation

        try:
            step_payload = {"id": env_id, "action": action_to_send}
            step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=timeout)
            step_res.raise_for_status()
            step_data = step_res.json()

            # Extract response data
            step_state = step_data["observation"]
            step_reward = step_data["reward"]
            step_done = step_data["done"]
            current_available_actions = step_data["available_actions"]
            current_observation = step_state
            
            # Store in memory (observation before action, and the action taken)
            memory.append({
                "observation": pre_observation,
                "action": action_to_send
            })
            
        except Exception as e:
            print(f"Step failed: {e}")
            step_reward = 0.0
            step_done = False

        # Update Loop State
        if step_done and step_reward > 0:
            solved = True

        if "Nothing happens" in step_state:
            invalid_count += 1
        
        done = step_done
        turn_number += 1
    
    train_reward = (10 if solved else 0.0) - 0.1 * float(invalid_count)
    
    return {
        "prompt_ids": episode_prompt_ids,
        "completion_ids": episode_completion_ids,
        "logprobs": episode_logprobs,
        "env_reward": train_reward
    }


def alfworld_rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30, history_length: int = 2) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import json
    
    # --- 1. Static Initialization (Once per Rank) ---
    # We check if the function has already established a connection for this worker
    if not getattr(alfworld_rollout_first_prompt_and_completion, "initialized", False):
        # Get local rank
        rank = int(os.environ.get("LOCAL_RANK", "0"))

        # Get env server for that local rank
        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]
        
        # Determine endpoint
        if not server_list:
            # Fallback (though likely fatal for the task)
            base_url = ""
            print("Warning: No ENVIRONMENT_SERVER_URLS found.")
        else:
            base_url = server_list[rank % len(server_list)]

        # Store endpoint on the function to avoid re-parsing
        alfworld_rollout_first_prompt_and_completion.base_url = base_url
        
        # Create environment (POST /create) - ONLY ONCE
        try:
            print(f"Initializing AlfWorld environment on rank {rank} at {base_url}...")
            create_res = requests.post(f"{base_url}/create", timeout=300)
            create_res.raise_for_status()
            # Store env_id on the function
            alfworld_rollout_first_prompt_and_completion.env_id = create_res.json()["id"]
            alfworld_rollout_first_prompt_and_completion.initialized = True
            print(f"Environment initialized. ID: {alfworld_rollout_first_prompt_and_completion.env_id}")
        except Exception as e:
            print(f"CRITICAL: Failed to create environment on rank {rank}: {e}")
            raise e

    # Retrieve static variables
    env_id = alfworld_rollout_first_prompt_and_completion.env_id
    env_endpoint = alfworld_rollout_first_prompt_and_completion.base_url

    # --- 2. Rollout Setup ---
    all_episode_prompt_ids: list[list[int]] = []
    all_episode_completion_ids: list[list[int]] = []
    all_episode_logprobs: list[list[float]] = []
    all_episode_rewards: list[float] = []

    tokenizer = trainer.processing_class
    DATA_LEN = 2500
    TIMEOUT = 2400

    # --- 3. Batch Loop ---
    for _ in prompts:
        # Randomize game for each episode
        game_id = random.randint(0, DATA_LEN - 1)
        
        episode = alfworld_rollout_once(
            trainer=trainer,
            env_id=env_id,
            env_endpoint=env_endpoint,
            game_id=game_id,
            max_turns=max_turns,
            history_length=history_length,
            timeout=TIMEOUT
        )
        
        all_episode_prompt_ids.append(episode["prompt_ids"])
        all_episode_completion_ids.append(episode["completion_ids"])
        all_episode_logprobs.append(episode["logprobs"])
        all_episode_rewards.append(episode["env_reward"])

    return {
        "prompt_ids": all_episode_prompt_ids,
        "completion_ids": all_episode_completion_ids,
        "logprobs": all_episode_logprobs,
        "env_rewards": all_episode_rewards
    }

def alfworld_rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)