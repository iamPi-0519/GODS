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


def alfworld_rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30, history_length: int = 2) -> dict[str, list]:
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
    game_ids = [random.randint(0, DATA_LEN - 1) for _ in range(num_episodes)]
    memories = [[] for _ in range(num_episodes)]
    task_descriptions = ["" for _ in range(num_episodes)]
    current_observations = ["" for _ in range(num_episodes)]
    current_actions = [[] for _ in range(num_episodes)]
    done_flags = [False for _ in range(num_episodes)]
    solved_flags = [False for _ in range(num_episodes)]
    invalid_counts = [0 for _ in range(num_episodes)]

    # Last turn data per episode (for backprop) - only store LAST turn's data
    last_turn_data_per_episode = [None for _ in range(num_episodes)]

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

            current_observations[i] = reset_data["observation"]
            current_actions[i] = reset_data["available_actions"]

            task_start = current_observations[i].find('Your task is to: ')
            if task_start != -1:
                task_descriptions[i] = current_observations[i][task_start + len('Your task is to: '):].strip()
            else:
                task_descriptions[i] = "complete the task"
        except Exception as e:
            print(f"Failed to reset game {game_ids[i]}: {e}")
            done_flags[i] = True

    # --- 5. Batched Turn Loop ---
    for turn in range(max_turns):
        active_indices = [i for i in range(num_episodes) if not done_flags[i]]
        if not active_indices:
            break

        # Build prompts for all active episodes
        batch_prompts = []
        for i in active_indices:
            if turn == 0 or history_length <= 0:
                reformatted_actions = "\n ".join(f"'{s}'" for s in current_actions[i] if s != 'help')
                formatted_obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=current_observations[i],
                    admissible_actions=reformatted_actions
                )
            else:
                reformatted_actions = "\n ".join(f"'{s}'" for s in current_actions[i] if s != 'help')
                recent = memories[i][-history_length:] if len(memories[i]) > history_length else memories[i]
                start_idx = len(memories[i]) - len(recent)

                lines = []
                for j, rec in enumerate(recent):
                    step_num = start_idx + j + 1
                    lines.append(f"[Observation {step_num}: '{rec['observation']}', Action {step_num}: '{rec['action']}']")
                action_history = "\n".join(lines)

                formatted_obs = ALFWORLD_TEMPLATE.format(
                    task_description=task_descriptions[i],
                    step_count=len(memories[i]),
                    history_length=len(recent),
                    action_history=action_history,
                    current_step=len(memories[i]) + 1,
                    current_observation=current_observations[i],
                    admissible_actions=reformatted_actions
                )

            messages = [{"role": "user", "content": formatted_obs}]
            prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            batch_prompts.append(prompt_text)

        # --- BATCHED GENERATION ---
        rollout_outputs = generate_rollout_completions(trainer, batch_prompts)

        # Process outputs and step environments (each episode uses its own env_id)
        for idx, i in enumerate(active_indices):
            output = rollout_outputs[idx]
            prompt_ids = output.get("prompt_ids", [])
            completion_ids = output.get("completion_ids", [])
            logprobs = output.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            # Store ONLY this turn's data
            last_turn_data_per_episode[i] = {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "logprobs": logprobs,
            }

            # Parse action
            action_to_send = completion_text
            if action_to_send.endswith("</s>"):
                action_to_send = action_to_send[:-5]
            if "<action>" in action_to_send.lower() and "</action>" in action_to_send.lower():
                action_start = action_to_send.lower().find("<action>") + len("<action>")
                action_end = action_to_send.lower().find("</action>")
                action_to_send = action_to_send[action_start:action_end].strip()

            pre_obs = current_observations[i]

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

                memories[i].append({"observation": pre_obs, "action": action_to_send})

                if step_done and step_reward > 0:
                    solved_flags[i] = True
                if "Nothing happens" in step_state:
                    invalid_counts[i] += 1
                done_flags[i] = step_done
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
    all_logprobs = []
    all_rewards = []

    for i in range(num_episodes):
        # Use ONLY last turn for backpropagation
        if last_turn_data_per_episode[i]:
            all_prompt_ids.append(last_turn_data_per_episode[i]["prompt_ids"])
            all_completion_ids.append(last_turn_data_per_episode[i]["completion_ids"])
            all_logprobs.append(last_turn_data_per_episode[i]["logprobs"])
        else:
            all_prompt_ids.append([])
            all_completion_ids.append([])
            all_logprobs.append([])

        train_reward = (10 if solved_flags[i] else 0.0) - 0.1 * float(invalid_counts[i])
        all_rewards.append(train_reward)

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_rewards": all_rewards
    }


def alfworld_rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)