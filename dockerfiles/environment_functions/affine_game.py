"""
Notes:
With the Affine GAME environment when you reset to start a new game you have to choose an 'opponent' type to train against.
Your two options are 'random' and 'mcts'.
Miners are free to choose which opponent type they train against.
"""

import re


def parse_scores_from_observation(obs: str) -> tuple[int, int] | None:
    """Extract (P0_score, P1_score) from an observation string.
    Returns None if the pattern isn't found (e.g., Game Over screen)."""
    match = re.search(r"Player 0:\s*(\d+)\s*points?,\s*Player 1:\s*(\d+)\s*points?", obs)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def parse_prize_card(obs: str) -> int | None:
    """Extract the current prize card value from an observation string."""
    match = re.search(r"Current point card:\s*(\d+)", obs)
    return int(match.group(1)) if match else None


def parse_final_outcome(obs: str) -> float:
    """Parse win/loss/draw from the Game Over observation.
    Returns +1.0 (win), -1.0 (loss), 0.0 (draw/unknown)."""
    if "Result: WIN" in obs:
        return 1.0
    if "Result: LOSS" in obs:
        return -1.0
    if "Result: DRAW" in obs or "Result: TIE" in obs:
        return 0.0
    # Fallback: parse "Your Return: X.X"
    match = re.search(r"Your Return:\s*([-\d.]+)", obs)
    if match:
        val = float(match.group(1))
        return 1.0 if val > 0 else (-1.0 if val < 0 else 0.0)
    return 0.0

GOOFSPIEL_SYSTEM_PROMPT = '''You are playing goofspiel.

# Game Rules
GOOFSPIEL RULES:
Setup: Each player has bid cards numbered 1 to N. A prize deck with cards 1 to N is shuffled.
Goal: Win the most points by bidding on prize cards.

Each turn:
1. Reveal top prize card (worth its face value in points)
2. Players simultaneously play one bid card from their hand
3. Highest bidder wins the prize card (adds its value to score)
4. If bids tie, prize card is discarded (no one gets points)

Winning: Player with most points after all rounds wins.


# Output Format
First, reason about your strategy inside <think> tags. Do NOT include any numbers or digits in your thinking - use only words.
Then, provide your action ID inside <answer> tags.

<think>strategy reasoning here, no numbers allowed</think><answer>ACTION_ID</answer>

Examples:
- For action "0 -> roll": <think>The prize is high and I should bid aggressively</think><answer>0</answer>
- For action "89 -> a3": <think>I should save my high cards for later</think><answer>89</answer>
'''.strip()


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

    # --- Helper: normalize Goofspiel observation and add Legal Actions ---
    def format_goofspiel_observation(raw_obs: str) -> str:
        """
        Parse the raw Goofspiel observation from the environment and return a
        normalized format that also includes a Legal Actions block for P0.
        If parsing fails, fall back to the original observation.
        """
        try:
            point_match = re.search(r"Current point card:\s*(\d+)", raw_obs)
            remaining_match = re.search(r"Remaining Point Cards:\s*([^\n]+)", raw_obs)
            p0_hand_match = re.search(r"P0 hand:\s*([^\n]+)", raw_obs)
            p1_hand_match = re.search(r"P1 hand:\s*([^\n]+)", raw_obs)
            win_seq_comment_match = re.search(r"\(Win sequence:[^\n]*\)", raw_obs)
            score_match = re.search(r"Player 0:\s*[^,\n]+,\s*Player 1:\s*[^\n]+", raw_obs)
            you_are_match = re.search(r"You are Player 0\.", raw_obs)

            if not (point_match and remaining_match and p0_hand_match and p1_hand_match and score_match and win_seq_comment_match):
                return raw_obs

            point_card = point_match.group(1).strip()
            remaining_cards = remaining_match.group(1).strip()
            p0_hand_str = p0_hand_match.group(1).strip()
            p1_hand_str = p1_hand_match.group(1).strip()
            win_seq_comment = win_seq_comment_match.group(0).strip()
            score_line = score_match.group(0).strip()
            you_are_line = you_are_match.group(0).strip() if you_are_match else "You are Player 0."

            # Build Legal Actions from P0 hand.
            p0_cards = [int(x) for x in re.findall(r"\d+", p0_hand_str)]
            p0_cards = sorted(set(p0_cards))
            legal_lines = [f"{card - 1} -> [P0]Bid: {card}" for card in p0_cards]

            lines = [
                f"Current point card: {point_card}",
                f"Remaining Point Cards: {remaining_cards}",
                f"P0 hand: {p0_hand_str}",
                f"P1 hand: {p1_hand_str}",
                "Win sequence:",
                win_seq_comment,
                score_line,
                "",
                "",
                you_are_line,
                "Legal Actions:",
                *legal_lines,
                "\nYour choice (ID only):"
            ]
            return "\n".join(lines)
        except Exception:
            # Never break training due to formatting issues
            return raw_obs

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
    game_ids = [
        random.randint(
            games_to_task_id_range[selected_game][0],
            games_to_task_id_range[selected_game][1]
        )
        for _ in range(num_episodes)
    ]
    episode_ids = [None] * num_episodes
    
    # --- 3. Per-Episode State Tracking ---
    current_observations = ["" for _ in range(num_episodes)]
    done_flags = [False for _ in range(num_episodes)]
    train_rewards = [0.0 for _ in range(num_episodes)]
    episode_lengths = [0 for _ in range(num_episodes)]
    
    # Multi-step accumulator variables (accumulate across all turns)
    accumulated_messages = [[] for _ in range(num_episodes)]
    episode_prompt_ids = [[] for _ in range(num_episodes)]
    episode_completion_ids = [[] for _ in range(num_episodes)]
    episode_logprobs = [[] for _ in range(num_episodes)]
    episode_action_masks = [[] for _ in range(num_episodes)]
    prev_full_ids = [None for _ in range(num_episodes)]

    # Per-turn game state tracking for reward computation
    prev_scores = [(0, 0)] * num_episodes                    # (p0, p1) scores before each turn
    episode_prize_cards = [[] for _ in range(num_episodes)]  # prize card values per turn
    episode_round_won = [[] for _ in range(num_episodes)]    # did P0 win each round?
    episode_final_obs = ["" for _ in range(num_episodes)]    # final observation for outcome parsing

    # --- Helper: log full episode turns (prompt/completion text) in plain text ---
    def log_episode_turns(episode_index: int) -> None:
        """
        For a single complete episode consisting of multiple turns, write out
        in plain text what the prompt text and completion text of each turn are
        to a log file.

        We treat each consecutive (user, assistant) pair in accumulated_messages
        as one turn:
          - Prompt text  = user message content
          - Completion   = assistant message content
        """
        try:
            messages = accumulated_messages[episode_index]
            if not messages:
                return

            log_path = os.environ.get(
                "AFFINE_GAME_EPISODE_LOG_PATH",
                "affine_game_episode_turns.log",
            )

            lines: list[str] = []
            lines.append(
                f"=== Episode index {episode_index}, episode_id={episode_ids[episode_index]} ==="
            )

            turn_id = 0
            i = 0
            # Walk sequentially through messages and pair user/assistant
            while i < len(messages):
                msg = messages[i]
                if (
                    msg.get("role") == "user"
                    and i + 1 < len(messages)
                    and messages[i + 1].get("role") == "assistant"
                ):
                    turn_id += 1
                    prompt_text = msg.get("content", "")
                    completion_text = messages[i + 1].get("content", "")

                    lines.append(f"Turn {turn_id} - Prompt:")
                    lines.append(prompt_text)
                    lines.append("Completion:")
                    lines.append(completion_text)
                    lines.append("---")
                    i += 2
                else:
                    i += 1

            lines.append("")  # trailing newline

            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            # Logging must be best-effort only; never break training
            print(
                f"[Affine GAME] Failed to write episode turn log for episode {episode_index}: {e}"
            )

    # --- 4. Reset All Games (Parallel) ---
    def reset_episode(i):
        payload = {"task_id": game_ids[i], "seed": i, "opponent": "mcts"}
        try:
            reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=TIMEOUT)
            reset_res.raise_for_status()
            reset_data = reset_res.json()
            result_block = reset_data["result"]
            
            # Store episode_id for this game instance
            episode_id = result_block.get("episode_id", "")
            
            # Get initial observation, normalize it, and append role + format instructions
            raw_observation = result_block.get("observation", "")
            base_observation = format_goofspiel_observation(raw_observation)
            observation = f"{GOOFSPIEL_SYSTEM_PROMPT}\n\n{base_observation}"
            
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

            # --- Per-turn episode logging ---
            print(f"[Episode {i}] Turn {turn}: prompt={obs_before[:120]}... | completion={completion_text} | action={action_to_send}")

            episode_data.append((i, completion_text, action_to_send, obs_before))
        
        # --- Step Environments (Parallel) ---
        def step_episode(i, completion_text, action_to_send, obs_before):
            try:
                step_payload = {"action": action_to_send, "episode_id": episode_ids[i]}
                step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=TIMEOUT)
                step_res.raise_for_status()
                step_data = step_res.json()
                step_block = step_data["result"]

                # Extract response data directly from the environment and normalize it
                raw_step_state = step_block.get("observation", "")
                step_state = format_goofspiel_observation(raw_step_state)
                step_reward = step_block.get("reward", 0)
                step_done = step_block.get("done", False)

                # Return the raw environment reward
                return i, completion_text, step_state, step_reward, step_done, None
            except Exception as e:
                print(f"Step failed for episode {i}: {e}")
                return i, completion_text, "", -0.01, True, e
        
        # Build obs_before lookup for per-turn reward tracking
        obs_before_map = {i: obs for i, _, _, obs in episode_data}

        # Execute steps in parallel
        with ThreadPoolExecutor(max_workers=min(len(episode_data), 10)) as executor:
            futures = [executor.submit(step_episode, i, comp_text, action, obs_before)
                      for i, comp_text, action, obs_before in episode_data]
            for future in as_completed(futures):
                i, completion_text, step_state, step_reward, step_done, error = future.result()

                current_observations[i] = step_state
                done_flags[i] = step_done

                # Track episode length for debugging/metrics.
                episode_lengths[i] += 1

                if step_done:
                    train_rewards[i] = step_reward

                # --- Per-turn tracking for reward computation ---
                obs_before_for_i = obs_before_map.get(i)
                if obs_before_for_i and not step_done:
                    prize = parse_prize_card(obs_before_for_i)
                    new_scores = parse_scores_from_observation(step_state)
                    if prize is not None and new_scores is not None:
                        p0_old, p1_old = prev_scores[i]
                        p0_new, p1_new = new_scores
                        p0_gained = p0_new - p0_old
                        episode_prize_cards[i].append(prize)
                        episode_round_won[i].append(p0_gained > 0)
                        prev_scores[i] = new_scores

                if step_done:
                    episode_final_obs[i] = step_state

                # Update messages for next turn
                assistant_msg = {"role": "assistant", "content": completion_text}
                accumulated_messages[i].append(assistant_msg)

                if not step_done:
                    user_msg = {"role": "user", "content": step_state}
                    accumulated_messages[i].append(user_msg)
                else:
                    log_episode_turns(i)

    # --- 6. Compute Reward Components ---
    outcome_rewards = []
    margin_rewards = []
    high_prize_rewards = []

    TOTAL_PRIZE_VALUE = 55  # 1+2+...+10 for 10-card Goofspiel
    HIGH_PRIZE_THRESHOLD = 6  # prizes >= 6 are "high value"
    HIGH_PRIZE_TOTAL = sum(range(HIGH_PRIZE_THRESHOLD, 11))  # 6+7+8+9+10 = 40

    # Win rate tracking
    wins = 0
    losses = 0
    draws = 0
    completed_games = 0

    for i in range(num_episodes):
        if not episode_prompt_ids[i] or not episode_completion_ids[i]:
            continue

        # 1. Outcome: parse from final observation
        outcome = parse_final_outcome(episode_final_obs[i])
        # Fallback: use API reward (1.0=win -> +1.0, 0.0=loss -> -1.0)
        if outcome == 0.0 and train_rewards[i] != 0.0:
            outcome = 1.0 if train_rewards[i] > 0 else -1.0
        outcome_rewards.append(outcome)

        # Track win/loss/draw counts
        if done_flags[i]:
            completed_games += 1
            if outcome > 0:
                wins += 1
            elif outcome < 0:
                losses += 1
            else:
                draws += 1

        # 2. Margin: point differential normalized by total prizes
        p0_final, p1_final = prev_scores[i]
        if p0_final == 0 and p1_final == 0:
            margin_rewards.append(outcome * 0.5)
        else:
            margin = (p0_final - p1_final) / TOTAL_PRIZE_VALUE
            margin = max(-1.0, min(1.0, margin))
            margin_rewards.append(margin)

        # 3. High-prize capture: fraction of high-value prizes won
        high_prizes_won = sum(
            prize for prize, won in zip(episode_prize_cards[i], episode_round_won[i])
            if won and prize >= HIGH_PRIZE_THRESHOLD
        )
        high_prize_rewards.append(high_prizes_won / HIGH_PRIZE_TOTAL if HIGH_PRIZE_TOTAL > 0 else 0.0)

    # --- Log win rate to WandB and stdout ---
    win_rate = wins / max(completed_games, 1)
    avg_margin = sum(margin_rewards) / max(len(margin_rewards), 1)
    avg_high_prize = sum(high_prize_rewards) / max(len(high_prize_rewards), 1)

    print(f"[Goofspiel] Batch stats: {wins}W/{losses}L/{draws}D "
          f"({completed_games} games, win_rate={win_rate:.2%}, "
          f"avg_margin={avg_margin:.3f}, avg_high_prize={avg_high_prize:.3f})")

    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                "game/win_rate": win_rate,
                "game/wins": wins,
                "game/losses": losses,
                "game/draws": draws,
                "game/completed_games": completed_games,
                "game/avg_margin": avg_margin,
                "game/avg_high_prize_capture": avg_high_prize,
            }, commit=False)
    except ImportError:
        pass

    # --- 7. Return Results ---
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
        "outcome_reward": outcome_rewards,
        "margin_reward": margin_rewards,
        "high_prize_reward": high_prize_rewards,
        "action_mask": all_action_masks,
    }

def rollout_reward_func(completions, **kwargs):
    """Legacy single-reward function. Kept for backward compatibility."""
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)


def reward_outcome(completions, **kwargs):
    """Win=+1, Draw=0, Loss=-1."""
    rewards = kwargs.get("outcome_reward")
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_margin(completions, **kwargs):
    """Normalized point differential (p0 - p1) / total_prizes. Range [-1, +1]."""
    rewards = kwargs.get("margin_reward")
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)


def reward_high_prize_capture(completions, **kwargs):
    """Fraction of high-value prizes (>=6) won. Range [0, 1]."""
    rewards = kwargs.get("high_prize_reward")
    return [float(r) for r in rewards] if rewards else [0.0] * len(completions)