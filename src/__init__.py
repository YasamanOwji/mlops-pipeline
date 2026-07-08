"""
generate_data.py
-----------------
Generates a synthetic but realistic dataset simulating student behavior
on a gamified learning platform (similar to Duolingo / Sololearn style apps).

Author: YasamanOwji
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

N = 3000  # number of students

def generate_dataset(n=N):
    df = pd.DataFrame()
    df["student_id"] = np.arange(1, n + 1)

    # Demographics / usage basics
    df["age"] = np.random.randint(13, 45, size=n)
    df["days_since_signup"] = np.random.randint(1, 720, size=n)

    # Gamification-related features
    df["current_streak_days"] = np.random.exponential(scale=5, size=n).astype(int)
    df["longest_streak_days"] = df["current_streak_days"] + np.random.randint(0, 30, size=n)
    df["total_xp"] = (df["days_since_signup"] * np.random.uniform(5, 40, size=n)).astype(int)
    df["league_tier"] = np.random.choice(
        ["Bronze", "Silver", "Gold", "Sapphire", "Diamond"],
        size=n,
        p=[0.35, 0.3, 0.2, 0.1, 0.05],
    )
    df["badges_earned"] = np.random.poisson(lam=3, size=n)
    df["daily_goal_completion_rate"] = np.clip(np.random.beta(2, 2, size=n), 0, 1)
    df["push_notifications_enabled"] = np.random.choice([0, 1], size=n, p=[0.3, 0.7])
    df["friends_count"] = np.random.poisson(lam=2, size=n)
    df["leaderboard_rank_percentile"] = np.random.uniform(0, 100, size=n)
    df["lessons_completed_last_30d"] = np.random.poisson(lam=12, size=n)
    df["avg_session_minutes"] = np.clip(np.random.normal(9, 4, size=n), 0.5, None)
    df["used_streak_freeze_last_30d"] = np.random.poisson(lam=0.8, size=n)
    df["reminder_notifications_last_30d"] = np.random.poisson(lam=6, size=n)

    # Build a realistic "engagement score" the target depends on
    engagement_score = (
        0.35 * (df["current_streak_days"] / (df["current_streak_days"].max() + 1))
        + 0.20 * df["daily_goal_completion_rate"]
        + 0.15 * (df["lessons_completed_last_30d"] / (df["lessons_completed_last_30d"].max() + 1))
        + 0.10 * df["push_notifications_enabled"]
        + 0.10 * (1 - df["leaderboard_rank_percentile"] / 100)
        + 0.10 * (df["friends_count"] / (df["friends_count"].max() + 1))
    )

    noise = np.random.normal(0, 0.03, size=n)
    engagement_score = np.clip(engagement_score + noise, 0, 1)

    midpoint = np.median(engagement_score)
    churn_prob = 1 / (1 + np.exp(12 * (engagement_score - midpoint)))
    churn_prob = np.clip(churn_prob + np.random.normal(0, 0.02, size=n), 0.02, 0.98)
    df["churned_next_30d"] = np.random.binomial(1, churn_prob)

    return df


if __name__ == "__main__":
    # ذخیره در مسیر درست نسبت به ریشه‌ی پروژه
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / "gamified_learning_engagement.csv"
    
    dataset = generate_dataset()
    dataset.to_csv(file_path, index=False)
    print(f"Dataset generated: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    print(dataset["churned_next_30d"].value_counts(normalize=True))