#python 4.2.1_Test_Size.py > 4.2.1_Test_Size.log 2>&1 &

import os
import numpy as np
import torch
import pandas as pd
import random
from concurrent.futures import ProcessPoolExecutor
from utils_evaluation import generate_embeddings, compute_similarity_scores, evaluate_eer_per_class

# ===================== Evaluation for subset of subjects =====================

def evaluate_for_subjects(num_subjects, x_test_e, y_test_e, x_test_v, y_test_v, distance="ed"):
    eers, stds = [], []
    unique_subjects = np.unique(y_test_e)

    for _ in range(50):
        selected_subjects = random.sample(list(unique_subjects), num_subjects)
        
        idx_e = np.isin(y_test_e, selected_subjects)
        idx_v = np.isin(y_test_v, selected_subjects)

        X_enroll, Y_enroll = x_test_e[idx_e], y_test_e[idx_e]
        X_verify, Y_verify = x_test_v[idx_v], y_test_v[idx_v]

        emb_enroll = generate_embeddings(X_enroll, None)
        emb_verify = generate_embeddings(X_verify, None)

        similarity_results = compute_similarity_scores(emb_enroll, Y_enroll, emb_verify, Y_verify, distance)
        avg_eer, std_eer = evaluate_eer_per_class(Y_enroll, similarity_results)

        eers.append(avg_eer)
        stds.append(std_eer)
        print(num_subjects, avg_eer, std_eer)

    return num_subjects, np.mean(eers), np.std(eers), np.mean(stds)

# ===================== Parallel Execution =====================

def run_evaluations(subject_range, x_test_e, y_test_e, x_test_v, y_test_v, distance="ed"):
    results = []
    random.shuffle(subject_range)

    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(evaluate_for_subjects, num, x_test_e, y_test_e, x_test_v, y_test_v, distance): num
            for num in subject_range
        }
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error for {futures[future]}: {e}")
    return results

# ===================== Entry Point =====================

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    distance = "ed"

    result_file = "./files/evaluation_results_g.csv"
    
    if os.path.exists(result_file):
        print("Results file found. Loading existing results...")
        df = pd.read_csv(result_file)
    else:
        x_test_e = np.load('./files/x_test_e.npy')
        y_test_e = np.load('./files/y_test_e.npy')
        x_test_v = np.load('./files/x_test_v.npy')
        y_test_v = np.load('./files/y_test_v.npy')

        subject_range = list(range(2, 101))
        results = run_evaluations(subject_range, x_test_e, y_test_e, x_test_v, y_test_v, distance)

        df = pd.DataFrame(results, columns=["NumSubjects", "AvgEER", "StdEER", "AvgStd"])
        df.to_csv(result_file, index=False)
        print(df)

    # ===================== Plotting =====================
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    plt.rc("font", size=14)
    plt.figure(figsize=(6.3, 4))
    sns.scatterplot(x="NumSubjects", y="AvgEER", data=df, color="green", s=80)
    sns.regplot(x="NumSubjects", y="AvgEER", data=df, scatter=False, color="blue", line_kws={"alpha": 0.5})
    plt.yticks(range(5, 13, 1))
    plt.xlabel("Number of Subjects")
    plt.ylabel("EER [%]")

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Average over 50 Trials'),
        Line2D([0], [0], color='blue', lw=2, alpha=0.5, label='EER Trendline')
    ]
    plt.legend(handles=legend_elements, loc="lower right", frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./Results/subject_number_eer.pdf", format="pdf")
    plt.show()
