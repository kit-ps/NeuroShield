# python 4.2.2_0_Session_Distribution.py > 4.2.2_0_Session_Distribution.log 2>&1 &

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prepare_data_for_evaluation(x_e, y_e, s_e, h_e, x_v, y_v, s_v, h_v):
    x_data = np.concatenate([x_e, x_v], axis=0)
    y_data = np.concatenate([y_e, y_v], axis=0)
    s_data = np.concatenate([s_e, s_v], axis=0)
    h_data = np.concatenate([h_e, h_v], axis=0)
    return x_data, y_data, s_data, h_data


def load_data():
    x_data, y_data, s_data, h_data = prepare_data_for_evaluation(
        np.load('./files/x_test_e.npy'), np.load('./files/y_test_e.npy'), np.load('./files/s_test_e.npy'), np.load('./files/h_test_e.npy'),
        np.load('./files/x_test_v.npy'), np.load('./files/y_test_v.npy'), np.load('./files/s_test_v.npy'), np.load('./files/h_test_v.npy')
    )
    return x_data, y_data, s_data, h_data


def analyze_sessions(y_data, h_data, s_data):
    # Normalize session to days since first session
    df_norm = pd.DataFrame({
        'subject_id': y_data,
        'hardware': h_data,
        'session': s_data
    })
    df_norm['session'] = df_norm.groupby(['subject_id', 'hardware'])['session'].transform(lambda x: x - x.min())

    # Print unique session info
    unique_sessions = df_norm.drop_duplicates(subset=['subject_id', 'hardware', 'session']).shape[0]
    print(f"Number of unique sessions: {unique_sessions}")

    print("\nUnique Sessions for Each Subject:")
    subj_sessions = df_norm.groupby(['subject_id', 'hardware'])['session'].unique()
    for subject_id, sessions in subj_sessions.items():
        print(f"Subject ID: {subject_id}, Unique Sessions: {sorted(sessions)}")


def plot_session_distribution(y_data, h_data, s_data):
    # Rebuild DataFrame with original session days
    df = pd.DataFrame({
        'subject_id': y_data,
        'hardware': h_data,
        'session': s_data
    })

    # Deduplicate to ensure unique triplets
    df = df.drop_duplicates(subset=['subject_id', 'hardware', 'session'])

    # Remove first session (day 0 or 1) and adjust
    df = df[df['session'] > 1]
    df['session'] = df['session'] - 1

    # Assign week number
    df['week'] = df['session'] // 7

    # Split into intervals
    df_weeks = df[df['week'] < 12].copy()
    df_months = df[(df['session'] >= 91.2501) & (df['session'] < 365.25)].copy()
    df_years = df[df['session'] >= 365.25].copy()

    # Compute month and year bins
    df_months['month'] = df_months['session'] // 30.4167
    df_years['year'] = df_years['session'] // 365.25

    # Count occurrences
    weekly_counts = df_weeks['week'].value_counts().sort_index()
    monthly_counts = df_months['month'].value_counts().sort_index()
    yearly_counts = df_years['year'].value_counts().sort_index()

    # Build labels exactly as original
    weekly_labels = [f'W{w+1}' for w in weekly_counts.index]
    monthly_labels = [f'M {i}' for i in range(4, 4 + len(monthly_counts))]
    yearly_labels = [f'Y {i}' for i in range(2, 2 + len(yearly_counts))]

    labels = weekly_labels + monthly_labels + yearly_labels
    counts = list(weekly_counts.values) + list(monthly_counts.values) + list(yearly_counts.values)
    colors = ['#4daf4a'] * len(weekly_labels) + ['#377eb8'] * len(monthly_labels) + ['#e41a1c'] * len(yearly_labels)

    # Plot
    plt.figure(figsize=(11.69, 3.5))
    bars = plt.bar(labels, counts, color=colors)
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), int(bar.get_height()),
                 va='bottom', ha='center', fontweight='bold', fontsize=12)

    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Time Interval of Sessions', fontsize=12)
    plt.ylabel('Number of Sessions', fontsize=12)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('unique_session_intervals.pdf', format='pdf', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Load data
    x_data, y_data, s_data, h_data = load_data()

    # Analyze and print unique session counts
    analyze_sessions(y_data, h_data, s_data)

    # Plot distribution matching original logic
    plot_session_distribution(y_data, h_data, s_data)