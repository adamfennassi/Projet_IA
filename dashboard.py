"""
Dashboard KPI pour comparer les performances des 3 algorithmes d'ordonnancement.
Genere des graphiques de comparaison et un tableau recapitulatif.

Usage:
    python dashboard.py          # utilise benchmark_results.json existant
    python dashboard.py --run    # relance le benchmark puis affiche
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non-interactif pour eviter le blocage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Se placer dans le dossier du script pour trouver les fichiers
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)


def load_or_run(force_run=False):
    if not force_run:
        try:
            with open(os.path.join(SCRIPT_DIR, "benchmark_results.json"), "r") as f:
                return json.load(f)
        except FileNotFoundError:
            pass

    from benchmark import run_benchmark
    return run_benchmark(num_test=10, num_train=200, seed=42)


def plot_dashboard(results):
    summary = results["summary"]
    training = results.get("training", {})
    raw = results.get("raw_results", {})

    diffs = list(summary.keys())
    methods = ["cpsat", "ml", "hybrid"]
    labels = ["CP-SAT (PPC)", "ML (PyTorch)", "Hybride (PPC+ML)"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Dashboard KPI - Comparaison des Algorithmes d'Ordonnancement\n"
        "CP-SAT (PPC) vs ML (PyTorch) vs Hybride (PPC+ML)",
        fontsize=15, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 3, hspace=0.45, wspace=0.35, top=0.91, bottom=0.08)
    x = np.arange(len(diffs))
    w = 0.25

    def get_vals(metric):
        out = []
        for m in methods:
            vals = []
            for d in diffs:
                v = summary[d].get(m, {}).get(metric)
                vals.append(v if v else 0)
            out.append(vals)
        return out

    # --- 1. Makespan moyen ---
    ax1 = fig.add_subplot(gs[0, 0])
    for i, (vals, lbl, c) in enumerate(zip(get_vals("avg_makespan"), labels, colors)):
        ax1.bar(x + i * w, vals, w, label=lbl, color=c, alpha=0.85)
    ax1.set_title("Makespan Moyen", fontweight="bold")
    ax1.set_xticks(x + w)
    ax1.set_xticklabels([d.capitalize() for d in diffs])
    ax1.set_ylabel("Makespan")
    ax1.legend(fontsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # --- 2. Temps de resolution ---
    ax2 = fig.add_subplot(gs[0, 1])
    for i, (vals, lbl, c) in enumerate(zip(get_vals("avg_solve_time"), labels, colors)):
        ax2.bar(x + i * w, vals, w, label=lbl, color=c, alpha=0.85)
    ax2.set_title("Temps de Resolution (s)", fontweight="bold")
    ax2.set_xticks(x + w)
    ax2.set_xticklabels([d.capitalize() for d in diffs])
    ax2.set_ylabel("Secondes")
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", alpha=0.3)

    # --- 3. Valeur objectif ---
    ax3 = fig.add_subplot(gs[0, 2])
    for i, (vals, lbl, c) in enumerate(zip(get_vals("avg_objective"), labels, colors)):
        ax3.bar(x + i * w, vals, w, label=lbl, color=c, alpha=0.85)
    ax3.set_title("Valeur Objectif (somme debuts)", fontweight="bold")
    ax3.set_xticks(x + w)
    ax3.set_xticklabels([d.capitalize() for d in diffs])
    ax3.set_ylabel("Objectif")
    ax3.legend(fontsize=7)
    ax3.grid(axis="y", alpha=0.3)

    # --- 4. Utilisation machines ---
    ax4 = fig.add_subplot(gs[1, 0])
    for i, (vals, lbl, c) in enumerate(zip(get_vals("avg_utilization"), labels, colors)):
        ax4.bar(x + i * w, vals, w, label=lbl, color=c, alpha=0.85)
    ax4.set_title("Utilisation Moyenne Machines (%)", fontweight="bold")
    ax4.set_xticks(x + w)
    ax4.set_xticklabels([d.capitalize() for d in diffs])
    ax4.set_ylabel("%")
    ax4.legend(fontsize=7)
    ax4.grid(axis="y", alpha=0.3)

    # --- 5. Taux de resolution ---
    ax5 = fig.add_subplot(gs[1, 1])
    for i, (m, lbl, c) in enumerate(zip(methods, labels, colors)):
        vals = []
        for d in diffs:
            s = summary[d].get(m, {}).get("solved", "0/0")
            parts = s.split("/")
            vals.append(int(parts[0]) / max(int(parts[1]), 1) * 100)
        ax5.bar(x + i * w, vals, w, label=lbl, color=c, alpha=0.85)
    ax5.set_title("Taux de Resolution (%)", fontweight="bold")
    ax5.set_xticks(x + w)
    ax5.set_xticklabels([d.capitalize() for d in diffs])
    ax5.set_ylabel("%")
    ax5.set_ylim(0, 110)
    ax5.legend(fontsize=7)
    ax5.grid(axis="y", alpha=0.3)

    # --- 6. Courbes d'entrainement ML ---
    ax6 = fig.add_subplot(gs[1, 2])
    loss_hist = training.get("loss_history", [])
    acc_hist = training.get("acc_history", [])
    if loss_hist:
        epochs = range(1, len(loss_hist) + 1)
        ax6_twin = ax6.twinx()
        l1, = ax6.plot(epochs, loss_hist, color="#FF9800", linewidth=2, label="Loss")
        l2, = ax6_twin.plot(epochs, [a * 100 for a in acc_hist], color="#4CAF50", linewidth=2, label="Precision %")
        ax6.set_xlabel("Epoque")
        ax6.set_ylabel("Loss", color="#FF9800")
        ax6_twin.set_ylabel("Precision (%)", color="#4CAF50")
        ax6.set_title("Entrainement ML", fontweight="bold")
        ax6.legend(handles=[l1, l2], fontsize=8, loc="center right")
        ax6.grid(alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "Pas de donnees\nd'entrainement", ha="center", va="center")
        ax6.set_title("Entrainement ML", fontweight="bold")

    # --- 7. Tableau recapitulatif ---
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")

    headers = ["Difficulte", "Methode", "Resolus", "Makespan", "Objectif", "Temps (s)", "Util. %"]
    rows = []
    for d in diffs:
        for m in methods:
            s = summary[d].get(m, {})
            rows.append([
                d.capitalize(),
                s.get("method", m),
                s.get("solved", "N/A"),
                f"{s['avg_makespan']:.1f}" if s.get("avg_makespan") else "N/A",
                f"{s['avg_objective']:.1f}" if s.get("avg_objective") else "N/A",
                f"{s['avg_solve_time']:.4f}" if s.get("avg_solve_time") else "N/A",
                f"{s['avg_utilization']:.1f}" if s.get("avg_utilization") else "N/A",
            ])

    table = ax7.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for j in range(len(headers)):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    color_map = {"CP-SAT": "#E3F2FD", "ML": "#FFF3E0", "Hybride": "#E8F5E9"}
    for i, row in enumerate(rows):
        bg = "#FFFFFF"
        for k, c in color_map.items():
            if k in row[1]:
                bg = c
                break
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(bg)

    ax7.set_title("Tableau Recapitulatif", fontweight="bold", fontsize=12, pad=20)

    plt.savefig("dashboard_kpi.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nDashboard sauvegarde : dashboard_kpi.png")


if __name__ == "__main__":
    force = "--run" in sys.argv

    print("=" * 60)
    print(" DASHBOARD KPI - ORDONNANCEMENT HYBRIDE PPC + ML")
    print("=" * 60)

    results = load_or_run(force_run=force)
    if results:
        plot_dashboard(results)
    else:
        print("Erreur lors du benchmark.")
