"""
Notes:
With the Affine GAME environment when you reset to start a new game you have to choose an 'opponent' type to train against.
Your two options are 'random' and 'mcts'.
Miners are free to choose which opponent type they train against. 
"""

GOOFSPIEL_FORMAT_INSTRUCTIONS = """You are playing Goofspiel (also known as the Game of Pure Strategy), a simultaneous-choice card game against an opponent.

Game Rules:
- You and your opponent each have cards numbered 1 through N (typically 13)
- A point card is revealed each turn, worth its face value in points
- Both players simultaneously choose one of their remaining cards to bid on the point card
- The player who plays the higher card wins the points
- If both players play the same card, the points are split
- Once a card is played, it cannot be used again
- The player with the most points at the end wins

Strategic Considerations:
- Consider the value of the current point card
- Think about which cards your opponent has already used
- Balance between winning high-value point cards and saving strong cards for later
- Your opponent is using MCTS (Monte Carlo Tree Search), so expect strategic play

Response Format:
When responding, you should first reason about your strategy, then select your action.
Your output must follow this format strictly:

Thought:
[Your strategic reasoning here - consider the point card value, remaining cards, opponent's likely strategy]

Action:
[A single number representing the card you want to play]

Example:
Thought:
The point card is worth 10 points, which is high value. I still have my 13, 12, and 11 cards. My opponent has used their 13 already. I should play my 12 to secure these points since I'm likely to win.

Action:
12"""

def rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # --- Constants for context length management ---
    MAX_EPISODE_TOKENS = 16384  # Max tokens for completion sequence (truncate if exceeded)
    MAX_PROMPT_LEN = 24576      # Max prompt tokens before ending episode early

    games_to_task_id_range = {
        "goofspiel": (0, 99999999),
        "liars_dice": (100000000, 199999999),
        "leduc_poker": (200000000, 299999999),
        "gin_rummy": (300000000, 399999999),
        "othello": (400000000, 499999999),
        "backgammon": (500000000, 599999999),
        "hex": (600000000, 699999999),
        "clobber": (700000000, 799999999),
    }

    selected_game = "goofspiel"
    
    # --- 1. Static Initialization (Once per Rank) ---
    # We check if the function has already established a connection for this worker
    if not getattr(rollout_first_prompt_and_completion, "initialized", False):
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
        rollout_first_prompt_and_completion.base_url = base_url
        rollout_first_prompt_and_completion.initialized = True
        print(f"Affine GAME endpoint initialized on rank {rank} at {base_url}")

    # Retrieve static variables
    env_endpoint = rollout_first_prompt_and_completion.base_url
    tokenizer = trainer.processing_class
    TIMEOUT = 2400
    num_episodes = len(prompts)

    # --- 2. Game ID Assignment---
    if trainer.model.training:
        num_generations = getattr(trainer, "num_generations", 1)
    else:
        num_generations = getattr(trainer, "num_generations_eval", getattr(trainer, "num_generations", 1))
    num_generations = max(1, int(num_generations))
    
    game_ids = []
    episode_ids = []
    current_group_game_id = None
    current_group_index = None
    for i in range(num_episodes):
        group_index = i // num_generations
        if group_index != current_group_index:
            current_group_index = group_index
            current_group_game_id = random.randint(games_to_task_id_range[selected_game][0], games_to_task_id_range[selected_game][1])
        game_ids.append(current_group_game_id)
        episode_ids.append(None)
    
    # --- 3. Per-Episode State Tracking ---
    current_observations = ["" for _ in range(num_episodes)]
    done_flags = [False for _ in range(num_episodes)]
    train_rewards = [0.0 for _ in range(num_episodes)]
    
    # Multi-step accumulator variables (accumulate across all turns)
    accumulated_messages = [[] for _ in range(num_episodes)]
    episode_prompt_ids = [[] for _ in range(num_episodes)]
    episode_completion_ids = [[] for _ in range(num_episodes)]
    episode_logprobs = [[] for _ in range(num_episodes)]
    episode_action_masks = [[] for _ in range(num_episodes)]
    prev_full_ids = [None for _ in range(num_episodes)]

    # --- 4. Reset All Games (Parallel) ---
    def reset_episode(i):
        payload = {"task_id": game_ids[i], "seed": 42, "opponent": "mcts"}
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]
            
            # Store episode_id for this game instance
            episode_id = result_block.get("episode_id", "")
            
            # Get initial observation and append format instructions (only in first turn)
            observation = result_block.get("observation", "") + GOOFSPIEL_FORMAT_INSTRUCTIONS
            
            # Initialize messages with first observation
            user_msg = {"role": "user", "content": observation}
            messages = [user_msg]
            
            return i, episode_id, observation, messages, False
        except Exception as e:
            print(f"Failed to reset game {game_ids[i]}: {e}")
            return i, "", "", [], True
    
    # Execute resets in parallel
    with ThreadPoolExecutor(max_workers=min(num_episodes, 10)) as executor:
        futures = [executor.submit(reset_episode, i) for i in range(num_episodes)]
        for future in as_completed(futures):
            i, episode_id, observation, messages, failed = future.result()
            episode_ids[i] = episode_id
            current_observations[i] = observation
            accumulated_messages[i] = messages
            done_flags[i] = failed

    # --- 5. Batched Turn Loop ---
    for turn in range(max_turns):
        active_indices = [i for i in range(num_episodes) if not done_flags[i]]
        if not active_indices:
            break

        # Build prompts for all active episodes
        batch_prompts = []
        for i in active_indices:
            batch_prompts.append(accumulated_messages[i])

        # --- BATCHED GENERATION ---
        # Generate completions for all active episodes at once
        rollout_outputs = generate_rollout_completions(trainer, prompts=batch_prompts, as_chat=True)

        # Process outputs - extract completions and parse actions
        episode_data = []
        for idx, i in enumerate(active_indices):
            output = rollout_outputs[idx]
            prompt_ids = output.get("prompt_ids", [])
            completion_ids = output.get("completion_ids", [])
            logprobs = output.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            # Check if prompt exceeds max length - end episode early to prevent context overflow
            if len(prompt_ids) > MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens ({len(prompt_ids)}) at turn {turn} for episode {i}, ending episode early")
                done_flags[i] = True
                continue

            # Multi-step accumulation logic (from alfworld_legacy.py)
            if turn == 0:
                episode_prompt_ids[i] = prompt_ids
                prev_full_ids[i] = prompt_ids.copy()
            else:
                if prev_full_ids[i] is None:
                    prev_full_ids[i] = prompt_ids.copy()
                elif prompt_ids[: len(prev_full_ids[i])] != prev_full_ids[i]:
                    # BPE mismatch - tokenizer produced different IDs for same prefix text
                    # Graceful fallback: skip delta masking for this turn, just add completion
                    print(
                        f"Warning: BPE mismatch at turn {turn} for episode {i} (expected prefix {len(prev_full_ids[i])}, "
                        f"got {len(prompt_ids)} tokens). Skipping delta mask for this turn."
                    )
                    # Reset prev_full_ids to current prompt to try to recover alignment
                    prev_full_ids[i] = prompt_ids.copy()
                else:
                    delta_prompt_ids = prompt_ids[len(prev_full_ids[i]) :]
                    if delta_prompt_ids:
                        episode_completion_ids[i].extend(delta_prompt_ids)
                        episode_logprobs[i].extend([0.0] * len(delta_prompt_ids))
                        episode_action_masks[i].extend([0] * len(delta_prompt_ids))
                    prev_full_ids[i] = prompt_ids.copy()

            if completion_ids:
                episode_completion_ids[i].extend(completion_ids)
                episode_logprobs[i].extend(logprobs)
                episode_action_masks[i].extend([1] * len(completion_ids))
                if prev_full_ids[i] is not None:
                    prev_full_ids[i] = prev_full_ids[i] + completion_ids

            # --- Parse Action ---
            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]

            # Parse ReAct format
            if "Action:" in action_to_send:
                action_to_send = action_to_send.split("Action:")[-1].strip()
            
            episode_data.append((i, completion_text, action_to_send))
        
        # --- Step Environments (Parallel) ---
        def step_episode(i, completion_text, action_to_send):
            try:
                step_payload = {"action": action_to_send, "episode_id": episode_ids[i]}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                # Extract response data
                step_state = step_block.get("observation", "")
                step_reward = step_block.get("reward", 0)
                step_done = step_block.get("done", False)
                
                return i, completion_text, step_state, step_reward, step_done, None
            except Exception as e:
                print(f"Step failed for episode {i}: {e}")
                return i, completion_text, "", -0.01, True, e
        
        # Execute steps in parallel
        with ThreadPoolExecutor(max_workers=min(len(episode_data), 10)) as executor:
            futures = [executor.submit(step_episode, i, comp_text, action) 
                      for i, comp_text, action in episode_data]
            for future in as_completed(futures):
                i, completion_text, step_state, step_reward, step_done, error = future.result()
                
                current_observations[i] = step_state
                done_flags[i] = step_done
                
                if step_done:
                    train_rewards[i] = step_reward

                # Update messages for next turn
                assistant_msg = {"role": "assistant", "content": completion_text}
                accumulated_messages[i].append(assistant_msg)

                if not step_done:
                    user_msg = {"role": "user", "content": step_state}
                    accumulated_messages[i].append(user_msg)

    # --- 6. Return Results ---
    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    all_rewards = []
    all_action_masks = []

    for i in range(num_episodes):
        # Skip episodes with empty sequences
        if not episode_prompt_ids[i] or not episode_completion_ids[i]:
            continue

        # Truncate episode if completion sequence exceeds max length
        if len(episode_completion_ids[i]) > MAX_EPISODE_TOKENS:
            print(f"Warning: Episode {i} completion exceeded {MAX_EPISODE_TOKENS} tokens ({len(episode_completion_ids[i])}), truncating")
            episode_completion_ids[i] = episode_completion_ids[i][:MAX_EPISODE_TOKENS]
            episode_logprobs[i] = episode_logprobs[i][:MAX_EPISODE_TOKENS]
            episode_action_masks[i] = episode_action_masks[i][:MAX_EPISODE_TOKENS]

        all_prompt_ids.append(episode_prompt_ids[i])
        all_completion_ids.append(episode_completion_ids[i])
        all_logprobs.append(episode_logprobs[i])
        all_rewards.append(train_rewards[i])
        all_action_masks.append(episode_action_masks[i])

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_rewards": all_rewards,
        "action_mask": all_action_masks
    }

def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)