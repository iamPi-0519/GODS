ALFWORLD_SYSTEM_PROMPT = """You are an expert agent operating in the ALFRED Embodied Environment. This is a text-based household simulation where you navigate rooms, locate objects, and manipulate them to complete tasks (e.g., "put some object in receptacle", "heat some object", "clean some object", "examine some object with light").

When responding:
1. You should first reason step-by-step about the current situation. When reasoning, you MUST put your thought within <think> </think> tags. This is especially helpful for complex decisions, planning multi-step actions, or when you need to analyze the current state.
2. Once you've finished your reasoning, you must provide your chosen action within <action> </action> tags.

Example:
<think>
I need to find an apple first. I should check common places where apples are found - countertops, fridge, or cabinets. Let me start by going to countertop 1 to look for an apple.
</think>
<action> go to countertop 1 </action>

Example without thinking (for straightforward actions):
<action> look </action>"""

ALFWORLD_OBSERVATION_TEMPLATE = """{observation}

Your admissible actions: [{admissible_actions}]"""


def alfworld_rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
    """
    Execute batched rollouts for ALFWorld environment.
    """
    from trl.experimental.openenv import generate_rollout_completions
    import os
    import random
    import requests

    # --- 1. Static Initialization (Once per Rank) ---
    if not getattr(alfworld_rollout_first_prompt_and_completion, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))

        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]

        if not server_list:
            base_url = ""
            print("Warning: No ENVIRONMENT_SERVER_URLS found.")
        else:
            base_url = server_list[rank % len(server_list)]

        alfworld_rollout_first_prompt_and_completion.base_url = base_url
        alfworld_rollout_first_prompt_and_completion.env_ids = []  # Will store multiple env_ids
        alfworld_rollout_first_prompt_and_completion.initialized = True
        print(f"AlfWorld endpoint initialized on rank {rank} at {base_url}")

    env_endpoint = alfworld_rollout_first_prompt_and_completion.base_url
    tokenizer = trainer.processing_class

    DATA_LEN = 2500
    TIMEOUT = 2400
    num_episodes = len(prompts)

    # --- 2. Create Environment Instances (one per episode) ---
    # Each episode needs its own env_id to maintain separate game state
    env_ids = []
    for i in range(num_episodes):
        try:
            create_res = requests.post(f"{env_endpoint}/create", timeout=300)
            create_res.raise_for_status()
            env_id = create_res.json()["id"]
            env_ids.append(env_id)
        except Exception as e:
            print(f"Failed to create environment for episode {i}: {e}")
            env_ids.append(None)

    # --- 3. Per-Episode State ---
    # Group episodes by num_generations to ensure each group attempts the same task
    if trainer.model.training:
        num_generations = getattr(trainer, "num_generations", 1)
    else:
        num_generations = getattr(trainer, "num_generations_eval", getattr(trainer, "num_generations", 1))
    num_generations = max(1, int(num_generations))
    
    game_ids = []
    current_group_game_id = None
    for i in range(num_episodes):
        if i // num_generations == 0:
            current_group_game_id = random.randint(0, DATA_LEN - 1)
        game_ids.append(current_group_game_id)
    current_observations = ["" for _ in range(num_episodes)]
    current_actions = [[] for _ in range(num_episodes)]
    done_flags = [False for _ in range(num_episodes)]
    solved_flags = [False for _ in range(num_episodes)]
    invalid_counts = [0 for _ in range(num_episodes)]

    # Multi-turn accumulator variables for training on all turns
    accumulated_messages = [[] for _ in range(num_episodes)]
    accumulated_completion_ids = [[] for _ in range(num_episodes)]
    accumulated_action_mask = [[] for _ in range(num_episodes)]
    accumulated_logprobs = [[] for _ in range(num_episodes)]
    initial_prompt_ids = [[] for _ in range(num_episodes)]

    # --- 4. Reset All Games (each with its own env_id) ---
    for i in range(num_episodes):
        if env_ids[i] is None:
            done_flags[i] = True
            continue

        payload = {"id": env_ids[i], "game": game_ids[i], "world_type": "Text"}
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()

            # DEBUG: Print response structure to diagnose missing 'observation' key
            print(f"[DEBUG] Reset response for game {game_ids[i]}: {list(reset_data.keys())}")
            if "observation" not in reset_data:
                print(f"[DEBUG] Full response: {reset_data}")

            current_observations[i] = reset_data["observation"]
            current_actions[i] = reset_data["available_actions"]

            # Initialize accumulated_messages with system prompt and initial observation
            system_msg = {"role": "system", "content": ALFWORLD_SYSTEM_PROMPT}
            
            # Format initial observation
            reformatted_actions = ", ".join(f"'{s}'" for s in current_actions[i] if s != 'help')
            initial_obs_content = ALFWORLD_OBSERVATION_TEMPLATE.format(
                observation=current_observations[i],
                admissible_actions=reformatted_actions
            )
            user_msg = {"role": "user", "content": initial_obs_content}
            accumulated_messages[i] = [system_msg, user_msg]
            
            # Tokenize initial prompt to get initial_prompt_ids
            prompt_text = tokenizer.apply_chat_template(
                accumulated_messages[i],
                add_generation_prompt=True,
                tokenize=False
            )
            initial_prompt_ids[i] = tokenizer.encode(prompt_text, add_special_tokens=False)
            
        except Exception as e:
            print(f"Failed to reset game {game_ids[i]}: {e}")
            done_flags[i] = True

    # --- 5. Batched Turn Loop ---
    for turn in range(max_turns):
        active_indices = [i for i in range(num_episodes) if not done_flags[i]]
        if not active_indices:
            break

        # Build prompts for all active episodes using accumulated_messages
        batch_prompts = []
        for i in active_indices:
            # Generate prompt from accumulated_messages
            prompt_text = tokenizer.apply_chat_template(
                accumulated_messages[i], 
                add_generation_prompt=True, 
                tokenize=False
            )
            batch_prompts.append(prompt_text)

        # --- BATCHED GENERATION ---
        generation_overrides = {"stop": ["</action>"], "include_stop_str_in_output": True}
        rollout_outputs = generate_rollout_completions(trainer, batch_prompts, generation_overrides=generation_overrides)

        # Process outputs and step environments (each episode uses its own env_id)
        for idx, i in enumerate(active_indices):
            output = rollout_outputs[idx]
            completion_text = tokenizer.decode(output["completion_ids"], skip_special_tokens=True).strip()

            # Parse action
            action_to_send = completion_text
            if "<action>" in action_to_send.lower() and "</action>" in action_to_send.lower():
                action_start = action_to_send.lower().find("<action>") + len("<action>")
                action_end = action_to_send.lower().find("</action>")
                action_to_send = action_to_send[action_start:action_end].strip()

            # Step environment using THIS episode's env_id
            try:
                step_payload = {"id": env_ids[i], "action": action_to_send}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()

                step_state = step_data["observation"]
                step_reward = step_data["reward"]
                step_done = step_data["done"]
                current_actions[i] = step_data["available_actions"]
                current_observations[i] = step_state

                if step_done and step_reward > 0:
                    solved_flags[i] = True
                if "Nothing happens" in step_state:
                    invalid_counts[i] += 1
                done_flags[i] = step_done
                
                # Add assistant message to accumulated_messages
                assistant_msg = {"role": "assistant", "content": completion_text}
                
                # Use model's actual completion_ids directly (no re-tokenization)
                # This guarantees perfect logprob alignment
                action_token_ids = output["completion_ids"]
                if isinstance(action_token_ids, list):
                    action_token_ids = list(action_token_ids)
                else:
                    action_token_ids = action_token_ids.tolist() if hasattr(action_token_ids, 'tolist') else list(action_token_ids)

                action_logprobs = output["logprobs"] if output.get("logprobs") else []
                if hasattr(action_logprobs, 'tolist'):
                    action_logprobs = action_logprobs.tolist()
                else:
                    action_logprobs = list(action_logprobs) if action_logprobs else []

                # Check if model's output ends with EOS token
                # If so, it's part of the action; if not, the observation will include closing tokens
                eos_token_id = tokenizer.eos_token_id
                model_generated_eos = action_token_ids and action_token_ids[-1] == eos_token_id

                # Accumulate action tokens with mask=1 and their true logprobs
                accumulated_completion_ids[i].extend(action_token_ids)
                accumulated_action_mask[i].extend([1] * len(action_token_ids))
                accumulated_logprobs[i].extend(action_logprobs)

                # Update messages for context tracking
                accumulated_messages[i].append(assistant_msg)

                if not step_done:
                    # Format next observation for user message
                    reformatted_actions = ", ".join(f"'{s}'" for s in current_actions[i] if s != 'help')
                    next_obs_content = ALFWORLD_OBSERVATION_TEMPLATE.format(
                        observation=current_observations[i],
                        admissible_actions=reformatted_actions
                    )
                    user_msg = {"role": "user", "content": next_obs_content}
                    accumulated_messages[i].append(user_msg)

                    # Compute observation tokens via delta tokenization
                    # Delta = from "end of assistant turn" to "ready for next generation"
                    # This gives us: [user_turn_tokens] [generation_prompt_tokens]
                    after_action_template = tokenizer.apply_chat_template(
                        accumulated_messages[i][:-1],  # messages with assistant
                        add_generation_prompt=False,
                        tokenize=False
                    )
                    next_gen_template = tokenizer.apply_chat_template(
                        accumulated_messages[i],  # messages with assistant + user
                        add_generation_prompt=True,
                        tokenize=False
                    )
                    obs_delta = next_gen_template[len(after_action_template):]
                    obs_token_ids = tokenizer.encode(obs_delta, add_special_tokens=False)

                    # If model didn't generate EOS, we need to prepend the closing token
                    # The chat template's after_action_template includes the closing,
                    # but our raw action tokens don't, so we need to add it
                    if not model_generated_eos and eos_token_id is not None:
                        obs_token_ids = [eos_token_id] + obs_token_ids

                    # Accumulate observation tokens with mask=0 and zero logprobs
                    accumulated_completion_ids[i].extend(obs_token_ids)
                    accumulated_action_mask[i].extend([0] * len(obs_token_ids))
                    accumulated_logprobs[i].extend([0.0] * len(obs_token_ids))
                
            except Exception as e:
                print(f"Step failed for episode {i}: {e}")
                done_flags[i] = True

    # --- 6. Cleanup ---
    for i in range(num_episodes):
        if env_ids[i] is not None:
            try:
                requests.post(f"{env_endpoint}/delete", json={"id": env_ids[i]}, timeout=60)
            except Exception:
                pass  # Best effort cleanup

    # --- 7. Return Results ---
    all_prompt_ids = []
    all_completion_ids = []
    all_action_mask = []
    all_logprobs = []
    all_rewards = []

    for i in range(num_episodes):
        # Skip episodes with empty sequences
        if not initial_prompt_ids[i] or not accumulated_completion_ids[i]:
            continue

        # Use accumulated data for all turns (multi-turn training)
        all_prompt_ids.append(initial_prompt_ids[i])
        all_completion_ids.append(accumulated_completion_ids[i])
        all_action_mask.append(accumulated_action_mask[i])
        all_logprobs.append(accumulated_logprobs[i])

        train_reward = (10 if solved_flags[i] else 0.0) - 0.1 * float(invalid_counts[i])
        all_rewards.append(train_reward)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_rewards": all_rewards,
        "action_mask": all_action_mask,
    }


def alfworld_rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)