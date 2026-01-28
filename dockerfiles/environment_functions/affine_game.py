"""
Notes:
With the Affine GAME environment when you reset to start a new game you have to choose an 'opponent' type to train against.
Your two options are 'random' and 'mcts'.
Miners are free to choose which opponent type they train against. 
"""

GOOFSPIEL_FORMAT_INSTRUCTIONS = """
You are Player 0 in Goofspiel.

Decision rule (STRICT):
- Read the line that starts with "Current point card:" to find the point card value, call it P.
- Check if card P is in your hand (look for "P0 hand: ..." and verify P appears in that list).
- If card P is in your hand, you MUST bid card P.
- Action IDs are 0-indexed: card value 1 -> action ID 0, card value 2 -> action ID 1, card value 3 -> action ID 2, etc.
- Therefore, to bid card P, output action ID = P - 1.

Output format (STRICT):
- Reply with ONLY the action ID as a single integer (no other text, labels, or numbers).
- Example: If point card is 3 and card 3 is in your hand, output "2" (because 3 - 1 = 2).
- Example: If point card is 11 and card 11 is in your hand, output "10" (because 11 - 1 = 10).
""".strip()


def rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests
    import json
    import re
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
    episode_total_rewards = [0.0 for _ in range(num_episodes)]
    episode_lengths = [0 for _ in range(num_episodes)]
    
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

            # --- Parse Action: first integer in the answer ---
            text = completion_text.strip()
            if text.endswith("</s>"):
                text = text[:-5].strip()
            match = re.search(r"\d+", text)
            action_to_send = match.group(0) if match else text
            
            # Keep track of observation used to choose this action for optional reward shaping
            obs_before = current_observations[i]

            # --- Debug logging: write model output and parsed action to file ---
            try:
                log_path = os.environ.get("AFFINE_GAME_LOG_PATH", "affine_game_model_outputs.log")

                # Try to infer the "correct" reference action from the observation, for easier debugging
                reference_action = None
                point_card = None
                point_match = re.search(r"Current point card:\s*(\d+)", obs_before)
                if point_match:
                    point_card = point_match.group(1)
                    point_card_int = int(point_card)
                    
                    # First, try to find Legal Actions block (for error prompts)
                    legal_section = obs_before
                    legal_start = obs_before.find("Legal Actions:")
                    if legal_start != -1:
                        legal_section = obs_before[legal_start:]
                        choice_idx = legal_section.find("Your choice")
                        if choice_idx != -1:
                            legal_section = legal_section[:choice_idx]
                        ref_pattern = rf"(\d+)\s*->\s*\[P0\]Bid:\s*{point_card}\b"
                        ref_match = re.search(ref_pattern, legal_section)
                        if ref_match:
                            reference_action = ref_match.group(1)
                    else:
                        # No Legal Actions block - compute from point card value directly
                        # Check if the point card is in P0's hand
                        p0_hand_match = re.search(r"P0 hand:\s*([^\n]+)", obs_before)
                        if p0_hand_match:
                            p0_hand_str = p0_hand_match.group(1)
                            # Check if point_card value is in the hand
                            hand_numbers = re.findall(r'\b\d+\b', p0_hand_str)
                            if point_card in hand_numbers:
                                # Action IDs are 0-indexed: card value 1 -> action 0, card value 2 -> action 1, etc.
                                reference_action = str(point_card_int - 1)

                log_record = {
                    "turn": int(turn),
                    "episode_index": int(i),
                    "action_to_send": action_to_send,
                    "reference_action": reference_action,
                    "point_card": point_card,
                    "completion_text": completion_text,
                    # Full observation text used to choose this action
                    "obs_before": obs_before,
                }
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_record) + "\n")
            except Exception as e:
                # Best-effort logging; never break training because of logging issues
                print(f"[Affine GAME] Failed to log model output for episode {i}, turn {turn}: {e}")

            episode_data.append((i, completion_text, action_to_send, obs_before))
        
        # --- Step Environments (Parallel) ---
        def step_episode(i, completion_text, action_to_send, obs_before):
            try:
                step_payload = {"action": action_to_send, "episode_id": episode_ids[i]}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                # Extract response data
                step_state = step_block.get("observation", "")
                step_reward = step_block.get("reward", 0)
                
                # Multiply base environment reward by 10x
                shaped_reward = step_reward * 10.0
                try:
                    reference_action = None
                    # 1) Parse current point card from the observation text
                    point_match = re.search(r"Current point card:\s*(\d+)", obs_before)
                    if point_match:
                        point_card = point_match.group(1)
                        point_card_int = int(point_card)
                        
                        # 2) First, try to find Legal Actions block (for error prompts from AgentGym)
                        legal_section = obs_before
                        legal_start = obs_before.find("Legal Actions:")
                        if legal_start != -1:
                            legal_section = obs_before[legal_start:]
                            choice_idx = legal_section.find("Your choice")
                            if choice_idx != -1:
                                legal_section = legal_section[:choice_idx]
                            # From legal actions, find the action id whose meaning is: "[P0]Bid: <point_card>"
                            ref_pattern = rf"(\d+)\s*->\s*\[P0\]Bid:\s*{point_card}\b"
                            ref_match = re.search(ref_pattern, legal_section)
                            if ref_match:
                                reference_action = ref_match.group(1)
                        else:
                            # No Legal Actions block - compute from point card value directly
                            # Check if the point card is in P0's hand
                            p0_hand_match = re.search(r"P0 hand:\s*([^\n]+)", obs_before)
                            if p0_hand_match:
                                p0_hand_str = p0_hand_match.group(1)
                                # Check if point_card value is in the hand
                                hand_numbers = re.findall(r'\b\d+\b', p0_hand_str)
                                if point_card in hand_numbers:
                                    # Action IDs are 0-indexed: card value 1 -> action 0, card value 2 -> action 1, etc.
                                    reference_action = str(point_card_int - 1)

                    # 3) If model's action matches this reference action id, add bonus.
                    if (
                        reference_action is not None
                        and isinstance(action_to_send, str)
                        and action_to_send.isdigit()
                        and action_to_send == reference_action
                    ):
                        shaped_reward += 1.0
                except Exception:
                    print(f"Warning: Failed to compute shaped reward for episode {i}")
                    shaped_reward = step_reward * 10.0
                
                step_done = step_block.get("done", False)
                
                return i, completion_text, step_state, shaped_reward, step_done, None
            except Exception as e:
                print(f"Step failed for episode {i}: {e}")
                return i, completion_text, "", -0.01, True, e
        
        # Execute steps in parallel
        with ThreadPoolExecutor(max_workers=min(len(episode_data), 10)) as executor:
            futures = [executor.submit(step_episode, i, comp_text, action, obs_before) 
                      for i, comp_text, action, obs_before in episode_data]
            for future in as_completed(futures):
                i, completion_text, step_state, step_reward, step_done, error = future.result()
                
                current_observations[i] = step_state
                done_flags[i] = step_done

                # Accumulate per-step rewards so we can compute a total over the whole episode.
                episode_total_rewards[i] += step_reward
                episode_lengths[i] += 1

                # When an episode finishes, store its total shaped reward in train_rewards.
                if step_done:
                    if episode_lengths[i] > 0:
                        train_rewards[i] = episode_total_rewards[i]
                    else:
                        # Fallback: if somehow length is zero, just use the last step reward
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