from areal.utils import stats_tracker


def filter_function(sample):
    # Use original_rewards if available for filtering and statistics
    rewards = (
        sample["original_rewards"]
        if "original_rewards" in sample
        else sample["rewards"]
    )
    reward0 = rewards[0]
    accept = any(r != reward0 for r in rewards)

    # Track rejection statistics
    if not accept:
        stats_tracker.get("rollout").scalar(rejected_by_failed_or_perfect=1)
        # Distinguish all-correct vs all-wrong
        all_positive = all(r > 0 for r in rewards)
        stats_tracker.get("rollout").scalar(rejected_by_all_correct=int(all_positive))
        stats_tracker.get("rollout").scalar(rejected_by_all_wrong=int(not all_positive))
    else:
        stats_tracker.get("rollout").scalar(rejected_by_failed_or_perfect=0)
        stats_tracker.get("rollout").scalar(rejected_by_all_correct=0)
        stats_tracker.get("rollout").scalar(rejected_by_all_wrong=0)

    return accept
