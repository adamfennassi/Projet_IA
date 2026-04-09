import os
import sys
import json
import io
import csv
import threading
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.core.files.base import ContentFile

from .models import BenchmarkResult
from .forms import CSVUploadForm

# Ajouter le dossier parent au path pour importer les algorithmes
PROJECT_DIR = str(settings.BASE_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from generator import SchedulingDatasetGenerator, taskInfo
from algorithms.cp_sat_solver import CPSatSolver
from algorithms.ml_solver import MLSolver
from algorithms.hybrid_solver import HybridSolver

_benchmark_lock = threading.Lock()
_benchmark_running = False


def _tasks_nt_to_dict(tasks):
    return {
        n: {"duration": t.duration, "predecessors": t.predecessors,
            "relase_date": t.relase_date, "due_date": t.due_date}
        for n, t in tasks.items()
    }


def index(request):
    results = BenchmarkResult.objects.all()[:10]
    return render(request, "webapp/index.html", {"results": results})


def upload_csv(request):
    if request.method == "POST":
        form = CSVUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES["csv_file"]
            content = csv_file.read().decode("utf-8")

            # Parser le CSV
            gen = SchedulingDatasetGenerator(seed=42)
            tasks = {}
            machines = []
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                if row["task_name"] == "MACHINES":
                    machines = row["duration"].split(",")
                    break
                if row["task_name"]:
                    tasks[row["task_name"]] = taskInfo(
                        duration=int(row["duration"]),
                        predecessors=row["predecessors"],
                        relase_date=int(row["relase_date"]),
                        due_date=int(row["due_date"]),
                    )

            tasks_dict = _tasks_nt_to_dict(tasks)

            # Resoudre avec CP-SAT uniquement
            solver = CPSatSolver(tasks_dict, machines, time_limit=30)
            sol = solver.solve()

            if not sol:
                return render(request, "webapp/upload.html", {
                    "form": form, "error": "Aucune solution trouvee pour ce dataset."
                })

            # Ajouter slack et infos complementaires
            for name, info in sol.items():
                td = tasks_dict[name]
                info["relase_date"] = td["relase_date"]
                info["due_date"] = td["due_date"]
                info["slack"] = td["due_date"] - info["end"]
                info["predecessors"] = td["predecessors"]

            gantt_buf = _generate_gantt_single(sol, machines, f"Resultat CP-SAT - {csv_file.name}")

            result = BenchmarkResult(name=f"CSV: {csv_file.name}")
            result.results_json = json.dumps({
                "kpis": _clean_kpis(solver.get_kpis()),
                "solution": sol,
            }, default=str)
            result.save()
            result.gantt_image.save("gantt.png", ContentFile(gantt_buf.getvalue()))

            return redirect("gantt", result_id=result.id)
    else:
        form = CSVUploadForm()

    return render(request, "webapp/upload.html", {"form": form})


def run_benchmark_view(request):
    global _benchmark_running

    if _benchmark_running:
        return render(request, "webapp/benchmark_form.html", {"running": True})

    if request.method == "POST":
        num_train = int(request.POST.get("num_train", 50))
        num_test = int(request.POST.get("num_test", 3))

        with _benchmark_lock:
            _benchmark_running = True

        try:
            gen = SchedulingDatasetGenerator(seed=42)

            # Entrainer le ML
            dummy_tasks, dummy_machines = gen.generate_dataset(**gen.MOYEN)
            dummy_dict = _tasks_nt_to_dict(dummy_tasks)
            ml_trainer = MLSolver(dummy_dict, dummy_machines)
            history = ml_trainer.train(gen, num_instances=num_train, epochs=30)

            if "error" in history:
                return render(request, "webapp/benchmark_form.html", {"error": history["error"]})

            trained_model = ml_trainer.model

            # Tester sur les 3 niveaux
            difficulties = {"facile": gen.FACILE, "moyen": gen.MOYEN, "difficile": gen.DIFFICILE}
            summary = {}

            for diff_name, config in difficulties.items():
                results_cpsat, results_ml, results_hybrid = [], [], []
                test_gen = SchedulingDatasetGenerator(seed=42 + hash(diff_name) % 1000)

                for _ in range(num_test):
                    tasks_nt, machines = test_gen.generate_dataset(**config)
                    td = _tasks_nt_to_dict(tasks_nt)

                    cp = CPSatSolver(td, machines, time_limit=30)
                    cp.solve()
                    results_cpsat.append(cp.get_kpis())

                    ml = MLSolver(td, machines, model=trained_model)
                    ml.solve()
                    results_ml.append(ml.get_kpis())

                    hy = HybridSolver(td, machines, ml_model=trained_model, time_limit=30)
                    hy.solve()
                    results_hybrid.append(hy.get_kpis())

                summary[diff_name] = {}
                for key, res_list in [("cpsat", results_cpsat), ("ml", results_ml), ("hybrid", results_hybrid)]:
                    solved = [r for r in res_list if r.get("makespan") is not None]
                    n = len(solved)
                    if n > 0:
                        summary[diff_name][key] = {
                            "method": solved[0]["method"],
                            "solved": f"{n}/{len(res_list)}",
                            "avg_makespan": round(np.mean([r["makespan"] for r in solved]), 1),
                            "avg_objective": round(np.mean([r["objective"] for r in solved]), 1),
                            "avg_solve_time": round(np.mean([r["solve_time"] for r in solved]), 4),
                            "avg_utilization": round(np.mean([r["avg_utilization"] for r in solved]), 1),
                        }
                    else:
                        summary[diff_name][key] = {
                            "method": res_list[0]["method"] if res_list else key,
                            "solved": f"0/{len(res_list)}",
                            "avg_makespan": None, "avg_objective": None,
                            "avg_solve_time": None, "avg_utilization": None,
                        }

            full_results = {
                "summary": summary,
                "training": {
                    "num_instances": num_train,
                    "training_time": ml_trainer.training_time,
                    "final_accuracy": history["machine_acc"][-1],
                    "loss_history": history["total_loss"],
                    "acc_history": history["machine_acc"],
                },
            }

            # Generer le dashboard
            dashboard_buf = _generate_dashboard_image(full_results)

            result = BenchmarkResult(name=f"Benchmark ({num_train} train, {num_test} test)")
            result.results_json = json.dumps(full_results, default=str)
            result.save()
            result.dashboard_image.save("dashboard.png", ContentFile(dashboard_buf.getvalue()))

            return redirect("dashboard")
        finally:
            with _benchmark_lock:
                _benchmark_running = False

    return render(request, "webapp/benchmark_form.html")


def dashboard_view(request):
    result = BenchmarkResult.objects.filter(dashboard_image__isnull=False).exclude(dashboard_image="").first()

    summary = None
    training = None
    if result:
        data = json.loads(result.results_json)
        summary = data.get("summary")
        training = data.get("training")

    return render(request, "webapp/dashboard.html", {
        "result": result,
        "summary": summary,
        "training": training,
    })


def gantt_view(request, result_id):
    result = get_object_or_404(BenchmarkResult, id=result_id)
    data = json.loads(result.results_json)

    # Grouper les taches par machine, triees par debut
    machine_tasks = {}
    if data.get("solution"):
        for task_name, info in data["solution"].items():
            m = info["machine"]
            if m not in machine_tasks:
                machine_tasks[m] = []
            machine_tasks[m].append({
                "name": task_name,
                "start": info["start"],
                "end": info["end"],
                "duration": info["duration"],
                "slack": info.get("slack", ""),
                "due_date": info.get("due_date", ""),
                "predecessors": info.get("predecessors", "none"),
            })
        for m in machine_tasks:
            machine_tasks[m].sort(key=lambda t: t["start"])

    return render(request, "webapp/gantt.html", {
        "result": result,
        "data": data,
        "machine_tasks": machine_tasks,
    })


def delete_result(request, result_id):
    if request.method == "POST":
        result = get_object_or_404(BenchmarkResult, id=result_id)
        # Supprimer les fichiers images associes
        if result.dashboard_image:
            result.dashboard_image.delete(save=False)
        if result.gantt_image:
            result.gantt_image.delete(save=False)
        result.delete()
    return redirect("index")


def delete_all(request):
    if request.method == "POST":
        for r in BenchmarkResult.objects.all():
            if r.dashboard_image:
                r.dashboard_image.delete(save=False)
            if r.gantt_image:
                r.gantt_image.delete(save=False)
        BenchmarkResult.objects.all().delete()
    return redirect("index")


# ---- Fonctions utilitaires ----

def _clean_kpis(kpis):
    """Nettoie les KPI pour la serialisation JSON."""
    return {k: v for k, v in kpis.items()}


def _generate_gantt_multi(solutions, machines, title="Diagramme de Gantt"):
    """Genere un Gantt comparatif avec plusieurs methodes."""
    project_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def get_project(name):
        parts = name.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else name

    all_tasks = set()
    for sol in solutions.values():
        all_tasks.update(sol.keys())
    projects = sorted(set(get_project(t) for t in all_tasks))
    proj_color = {p: project_colors[i % 10] for i, p in enumerate(projects)}

    n = len(solutions)
    fig, axes = plt.subplots(n, 1, figsize=(14, max(3, len(machines) * 1.2) * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    for ax, (method_name, sol) in zip(axes, solutions.items()):
        for m_idx, m in enumerate(machines):
            tasks_on_m = [(nm, s) for nm, s in sol.items() if s["machine"] == m]
            tasks_on_m.sort(key=lambda x: x[1]["start"])
            for name, info in tasks_on_m:
                color = proj_color[get_project(name)]
                ax.barh(m_idx, info["duration"], left=info["start"], height=0.6,
                        color=color, edgecolor="black", linewidth=1, alpha=0.85)
                ax.text(info["start"] + info["duration"] / 2, m_idx, name,
                        ha="center", va="center", fontsize=7, fontweight="bold",
                        color="white", bbox=dict(boxstyle="round,pad=0.2",
                        facecolor="black", alpha=0.3, edgecolor="none"))

        makespan = max(s["end"] for s in sol.values())
        obj = sum(s["start"] for s in sol.values())
        ax.set_yticks(range(len(machines)))
        ax.set_yticklabels(machines, fontweight="bold")
        ax.set_title(f"{method_name}  |  Makespan = {makespan}  |  Objectif = {obj:.0f}",
                     fontweight="bold", fontsize=11)
        ax.grid(axis="x", alpha=0.3, linestyle="--")

    axes[-1].set_xlabel("Temps")

    legend = [mpatches.Patch(color=proj_color[p], label=p) for p in projects]
    fig.legend(handles=legend, loc="lower center", ncol=min(len(projects), 8), fontsize=8)

    plt.tight_layout(rect=[0, 0.04, 1, 0.94])
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


def _generate_gantt_single(solution, machines, title="Diagramme de Gantt"):
    project_colors = plt.cm.tab10(np.linspace(0, 1, 10))

    def get_project(name):
        parts = name.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else name

    projects = sorted(set(get_project(t) for t in solution))
    proj_color = {p: project_colors[i % 10] for i, p in enumerate(projects)}

    fig, ax = plt.subplots(figsize=(14, max(4, len(machines) * 1.2)))

    for m_idx, m in enumerate(machines):
        tasks_on_m = [(n, s) for n, s in solution.items() if s["machine"] == m]
        tasks_on_m.sort(key=lambda x: x[1]["start"])
        for name, info in tasks_on_m:
            color = proj_color[get_project(name)]
            ax.barh(m_idx, info["duration"], left=info["start"], height=0.6,
                    color=color, edgecolor="black", linewidth=1, alpha=0.85)
            ax.text(info["start"] + info["duration"] / 2, m_idx, name,
                    ha="center", va="center", fontsize=7, fontweight="bold",
                    color="white", bbox=dict(boxstyle="round,pad=0.2",
                    facecolor="black", alpha=0.3, edgecolor="none"))

    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels(machines, fontweight="bold")
    ax.set_xlabel("Temps")
    ax.set_title(title, fontweight="bold", fontsize=13)
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    legend = [mpatches.Patch(color=proj_color[p], label=p) for p in projects]
    ax.legend(handles=legend, loc="upper right", fontsize=8)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf


def _generate_dashboard_image(results):
    summary = results["summary"]
    training = results.get("training", {})

    diffs = list(summary.keys())
    methods = ["cpsat", "ml", "hybrid"]
    labels = ["CP-SAT (PPC)", "ML (PyTorch)", "Hybride (PPC+ML)"]
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Dashboard KPI - CP-SAT vs ML vs Hybride", fontsize=15, fontweight="bold", y=0.98)

    x = np.arange(len(diffs))
    w = 0.25

    def vals_for(metric):
        out = []
        for m in methods:
            v = []
            for d in diffs:
                val = summary[d].get(m, {}).get(metric)
                v.append(val if val else 0)
            out.append(v)
        return out

    specs = [
        (231, "Makespan Moyen", "avg_makespan"),
        (232, "Temps de Resolution (s)", "avg_solve_time"),
        (233, "Valeur Objectif", "avg_objective"),
        (234, "Utilisation Machines (%)", "avg_utilization"),
    ]

    for pos, title, metric in specs:
        ax = fig.add_subplot(pos)
        for i, (v, lbl, c) in enumerate(zip(vals_for(metric), labels, colors)):
            ax.bar(x + i * w, v, w, label=lbl, color=c, alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x + w)
        ax.set_xticklabels([d.capitalize() for d in diffs])
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)

    # Courbe d'entrainement
    ax5 = fig.add_subplot(235)
    loss = training.get("loss_history", [])
    acc = training.get("acc_history", [])
    if loss:
        ep = range(1, len(loss) + 1)
        ax5t = ax5.twinx()
        ax5.plot(ep, loss, color="#FF9800", linewidth=2, label="Loss")
        ax5t.plot(ep, [a * 100 for a in acc], color="#4CAF50", linewidth=2, label="Precision %")
        ax5.set_xlabel("Epoque")
        ax5.set_ylabel("Loss", color="#FF9800")
        ax5t.set_ylabel("Precision (%)", color="#4CAF50")
        ax5.set_title("Entrainement ML", fontweight="bold")
        ax5.grid(alpha=0.3)

    # Taux de resolution
    ax6 = fig.add_subplot(236)
    for i, (m, lbl, c) in enumerate(zip(methods, labels, colors)):
        v = []
        for d in diffs:
            s = summary[d].get(m, {}).get("solved", "0/0")
            parts = s.split("/")
            v.append(int(parts[0]) / max(int(parts[1]), 1) * 100)
        ax6.bar(x + i * w, v, w, label=lbl, color=c, alpha=0.85)
    ax6.set_title("Taux de Resolution (%)", fontweight="bold")
    ax6.set_xticks(x + w)
    ax6.set_xticklabels([d.capitalize() for d in diffs])
    ax6.set_ylim(0, 110)
    ax6.legend(fontsize=7)
    ax6.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return buf
