"""
Benchmark comparant les 3 approches d'ordonnancement :
  1. CP-SAT (PPC pure)
  2. ML (PyTorch pur)
  3. Hybride (PPC + ML)

Genere des instances, resout avec chaque methode, collecte les KPI.
"""

import sys
import os
import json
import numpy as np

# Se placer dans le dossier du script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from generator import SchedulingDatasetGenerator
from algorithms.cp_sat_solver import CPSatSolver
from algorithms.ml_solver import MLSolver
from algorithms.hybrid_solver import HybridSolver


def tasks_nt_to_dict(tasks):
    return {
        n: {"duration": t.duration, "predecessors": t.predecessors,
            "relase_date": t.relase_date, "due_date": t.due_date}
        for n, t in tasks.items()
    }


def run_benchmark(num_test=10, num_train=200, seed=42):
    print("=" * 70)
    print(" BENCHMARK : CP-SAT vs ML vs HYBRIDE")
    print("=" * 70)

    # --- Phase 1 : Entrainement ML ---
    print(f"\n[1/3] Entrainement ML sur {num_train} instances...")
    train_gen = SchedulingDatasetGenerator(seed=seed)
    dummy_tasks, dummy_machines = train_gen.generate_dataset(**train_gen.MOYEN)

    ml_trainer = MLSolver(tasks_nt_to_dict(dummy_tasks), dummy_machines)
    history = ml_trainer.train(train_gen, num_instances=num_train, epochs=100)

    if "error" in history:
        print(f"ERREUR: {history['error']}")
        return None

    print(f"   Temps d'entrainement: {ml_trainer.training_time:.2f}s")
    print(f"   Precision finale: {history['machine_acc'][-1]:.2%}")

    trained_model = ml_trainer.model

    # --- Phase 2 : Evaluation ---
    print(f"\n[2/3] Evaluation sur {num_test} instances par difficulte...\n")

    difficulties = {
        "facile": SchedulingDatasetGenerator.FACILE,
        "moyen": SchedulingDatasetGenerator.MOYEN,
        "difficile": SchedulingDatasetGenerator.DIFFICILE,
    }

    all_results = {}

    for diff_name, config in difficulties.items():
        print(f"   --- {diff_name.upper()} ---")
        res = {"cpsat": [], "ml": [], "hybrid": []}
        test_gen = SchedulingDatasetGenerator(seed=seed + hash(diff_name) % 1000)

        for i in range(num_test):
            tasks_nt, machines = test_gen.generate_dataset(**config)
            td = tasks_nt_to_dict(tasks_nt)

            cpsat = CPSatSolver(td, machines, time_limit=30)
            cpsat.solve()
            res["cpsat"].append(cpsat.get_kpis())

            ml = MLSolver(td, machines, model=trained_model)
            ml.solve()
            res["ml"].append(ml.get_kpis())

            hybrid = HybridSolver(td, machines, ml_model=trained_model, time_limit=30)
            hybrid.solve()
            res["hybrid"].append(hybrid.get_kpis())

            sys.stdout.write(f"\r   Instance {i+1}/{num_test}")
            sys.stdout.flush()

        print()
        all_results[diff_name] = res

    # --- Phase 3 : Aggregation ---
    print(f"\n[3/3] Aggregation...\n")

    summary = {}
    for diff, methods in all_results.items():
        summary[diff] = {}
        for key, results in methods.items():
            solved = [r for r in results if r.get("makespan") is not None]
            n = len(results)
            if solved:
                summary[diff][key] = {
                    "method": solved[0]["method"],
                    "solved": f"{len(solved)}/{n}",
                    "avg_makespan": round(np.mean([r["makespan"] for r in solved]), 2),
                    "avg_objective": round(np.mean([r["objective"] for r in solved]), 2),
                    "avg_solve_time": round(np.mean([r["solve_time"] for r in solved]), 4),
                    "avg_utilization": round(np.mean([r["avg_utilization"] for r in solved]), 2),
                }
            else:
                summary[diff][key] = {
                    "method": results[0]["method"] if results else key,
                    "solved": f"0/{n}",
                    "avg_makespan": None, "avg_objective": None,
                    "avg_solve_time": None, "avg_utilization": None,
                }

    # Affichage
    print("=" * 90)
    print(" RESULTATS")
    print("=" * 90)

    for diff, methods in summary.items():
        print(f"\n  {diff.upper()}")
        print(f"  {'Methode':<22} {'Resolus':<10} {'Makespan':<12} {'Objectif':<12} {'Temps(s)':<12} {'Util.%':<10}")
        print(f"  {'─'*78}")
        for stats in methods.values():
            nm = stats["method"]
            sv = stats["solved"]
            mk = f"{stats['avg_makespan']:.1f}" if stats["avg_makespan"] else "N/A"
            ob = f"{stats['avg_objective']:.1f}" if stats["avg_objective"] else "N/A"
            t = f"{stats['avg_solve_time']:.4f}" if stats["avg_solve_time"] else "N/A"
            u = f"{stats['avg_utilization']:.1f}" if stats["avg_utilization"] else "N/A"
            print(f"  {nm:<22} {sv:<10} {mk:<12} {ob:<12} {t:<12} {u:<10}")

    # Sauvegarder
    output = {
        "summary": summary,
        "raw_results": {
            d: {m: [{k: v for k, v in r.items() if k != "machine_utilization"} for r in rs]
                for m, rs in methods.items()}
            for d, methods in all_results.items()
        },
        "training": {
            "num_instances": num_train,
            "training_time": ml_trainer.training_time,
            "final_accuracy": history["machine_acc"][-1],
            "final_loss": history["total_loss"][-1],
            "loss_history": history["total_loss"],
            "acc_history": history["machine_acc"],
        },
    }

    with open("benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResultats sauvegardes dans benchmark_results.json")
    return output


if __name__ == "__main__":
    run_benchmark(num_test=3, num_train=50, seed=42)
