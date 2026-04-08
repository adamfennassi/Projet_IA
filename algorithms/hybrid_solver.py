"""
Solveur Hybride PPC + ML pour l'ordonnancement sur machines paralleles.

Strategie :
1. Le ML predit une solution initiale (affectation + priorites)
2. Ces predictions sont injectees comme HINTS dans CP-SAT
3. CP-SAT demarre pres de l'optimum et affine la solution exacte
"""

import time
import torch
from ortools.sat.python import cp_model

from .ml_solver import _extract_features


class HybridSolver:

    def __init__(self, tasks, machines, ml_model=None, time_limit=30):
        self.tasks = tasks
        self.machines = machines
        self.ml_model = ml_model
        self.time_limit = time_limit
        self.solution = None
        self.solve_time = None
        self.ml_predict_time = None
        self.cpsat_time = None
        self.status = None
        self.objective_value = None
        self.hints_used = False

    def _get_ml_hints(self):
        """Utilise le ML pour generer des hints (solution initiale) pour CP-SAT."""
        if self.ml_model is None:
            return None

        self.ml_model.eval()
        features = _extract_features(self.tasks, self.machines)
        names = list(self.tasks.keys())
        X = torch.tensor([features[n] for n in names], dtype=torch.float32)

        with torch.no_grad():
            logits, priorities = self.ml_model(X)

        machine_preds = logits.argmax(1).numpy()
        prios = priorities.numpy()

        # Construire une solution par dispatching (meme logique que MLSolver)
        order = sorted(range(len(names)), key=lambda i: prios[i])
        machine_end = {m: 0 for m in self.machines}
        task_ends = {}
        hints = {}

        for idx in order:
            name = names[idx]
            info = self.tasks[name]

            earliest = info["relase_date"]
            pred = info["predecessors"]
            if pred != "none" and pred in task_ends:
                earliest = max(earliest, task_ends[pred])

            pred_m = machine_preds[idx]
            preferred = self.machines[pred_m] if pred_m < len(self.machines) else self.machines[0]

            best_m = preferred
            best_start = max(earliest, machine_end[preferred])
            for m in self.machines:
                cand = max(earliest, machine_end[m])
                if cand < best_start:
                    best_start = cand
                    best_m = m

            hints[name] = {"start_hint": best_start, "machine_hint": best_m}
            machine_end[best_m] = best_start + info["duration"]
            task_ends[name] = best_start + info["duration"]

        return hints

    def solve(self):
        """
        Resolution hybride : ML hints + CP-SAT exact.
        Returns: dict solution ou None
        """
        t0 = time.time()

        # Phase 1 : predictions ML
        ml_t0 = time.time()
        hints = self._get_ml_hints()
        self.ml_predict_time = time.time() - ml_t0
        self.hints_used = hints is not None

        # Phase 2 : CP-SAT avec hints
        cpsat_t0 = time.time()
        model = cp_model.CpModel()

        start_vars = {}
        for name, info in self.tasks.items():
            lb = info["relase_date"]
            ub = info["due_date"] - info["duration"]
            if ub < lb:
                self.solve_time = time.time() - t0
                self.cpsat_time = time.time() - cpsat_t0
                self.status = "INFEASIBLE"
                return None
            start_vars[name] = model.new_int_var(lb, ub, f"start_{name}")

        machine_vars = {
            name: {m: model.new_bool_var(f"{name}_on_{m}") for m in self.machines}
            for name in self.tasks
        }

        interval_vars = {}
        for name, info in self.tasks.items():
            interval_vars[name] = {}
            for m in self.machines:
                interval_vars[name][m] = model.new_optional_fixed_size_interval_var(
                    start=start_vars[name],
                    size=info["duration"],
                    is_present=machine_vars[name][m],
                    name=f"interval_{name}_on_{m}",
                )

        for name in self.tasks:
            model.add_exactly_one(machine_vars[name].values())

        for m in self.machines:
            model.add_no_overlap([interval_vars[name][m] for name in self.tasks])

        for name, info in self.tasks.items():
            pred = info["predecessors"]
            if pred != "none":
                model.Add(start_vars[name] >= start_vars[pred] + self.tasks[pred]["duration"])

        model.Minimize(sum(start_vars.values()))

        # Injecter les hints ML
        if hints:
            for name, hint in hints.items():
                model.AddHint(start_vars[name], hint["start_hint"])
                for m in self.machines:
                    model.AddHint(machine_vars[name][m], 1 if m == hint["machine_hint"] else 0)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        status = solver.solve(model)

        self.cpsat_time = time.time() - cpsat_t0
        self.solve_time = time.time() - t0

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            self.status = "OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE"
            self.objective_value = solver.objective_value
            self.solution = {}
            for name, info in self.tasks.items():
                st = solver.value(start_vars[name])
                machine = next(m for m in self.machines if solver.value(machine_vars[name][m]))
                self.solution[name] = {
                    "start": st,
                    "end": st + info["duration"],
                    "duration": info["duration"],
                    "machine": machine,
                }
            return self.solution
        else:
            self.status = "INFEASIBLE"
            return None

    def get_makespan(self):
        if not self.solution:
            return None
        return max(v["end"] for v in self.solution.values())

    def get_kpis(self):
        if not self.solution:
            return {
                "method": "Hybride (PPC+ML)",
                "status": self.status,
                "solve_time": self.solve_time,
                "ml_predict_time": self.ml_predict_time,
                "cpsat_time": self.cpsat_time,
                "hints_used": self.hints_used,
                "makespan": None,
                "objective": None,
                "avg_utilization": None,
            }

        makespan = self.get_makespan()
        min_start = min(v["start"] for v in self.solution.values())
        span = max(makespan - min_start, 1)
        util = {}
        for m in self.machines:
            work = sum(v["duration"] for v in self.solution.values() if v["machine"] == m)
            util[m] = work / span * 100
        avg_util = sum(util.values()) / len(util)

        return {
            "method": "Hybride (PPC+ML)",
            "status": self.status,
            "solve_time": self.solve_time,
            "ml_predict_time": self.ml_predict_time,
            "cpsat_time": self.cpsat_time,
            "hints_used": self.hints_used,
            "makespan": makespan,
            "objective": self.objective_value,
            "avg_utilization": round(avg_util, 2),
            "machine_utilization": util,
        }
