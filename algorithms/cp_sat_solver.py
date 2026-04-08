"""
Solveur CP-SAT pur (Programmation Par Contraintes) pour l'ordonnancement
sur machines paralleles. Utilise Google OR-Tools.
"""

import time
from ortools.sat.python import cp_model


class CPSatSolver:

    def __init__(self, tasks, machines, time_limit=30):
        """
        Args:
            tasks: dict {nom: {duration, predecessors, relase_date, due_date}}
            machines: list de noms de machines
            time_limit: limite de temps du solveur en secondes
        """
        self.tasks = tasks
        self.machines = machines
        self.time_limit = time_limit
        self.solution = None
        self.solve_time = None
        self.status = None
        self.objective_value = None

    def solve(self):
        """
        Resout le probleme avec CP-SAT.
        Returns: dict {task_name: {start, end, duration, machine}} ou None
        """
        t0 = time.time()
        model = cp_model.CpModel()

        # Variables de debut
        start_vars = {}
        for name, info in self.tasks.items():
            lb = info["relase_date"]
            ub = info["due_date"] - info["duration"]
            if ub < lb:
                self.solve_time = time.time() - t0
                self.status = "INFEASIBLE"
                return None
            start_vars[name] = model.new_int_var(lb, ub, f"start_{name}")

        # Affectation tache -> machine
        machine_vars = {
            name: {m: model.new_bool_var(f"{name}_on_{m}") for m in self.machines}
            for name in self.tasks
        }

        # Intervalles optionnels pour non-chevauchement
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

        # Chaque tache sur exactement 1 machine
        for name in self.tasks:
            model.add_exactly_one(machine_vars[name].values())

        # Non-chevauchement par machine
        for m in self.machines:
            model.add_no_overlap([interval_vars[name][m] for name in self.tasks])

        # Precedence
        for name, info in self.tasks.items():
            pred = info["predecessors"]
            if pred != "none":
                # La tache courante commence apres la fin de son predecesseur
                model.Add(start_vars[name] >= start_vars[pred] + self.tasks[pred]["duration"])

        # Objectif : minimiser la somme des dates de debut
        model.Minimize(sum(start_vars.values()))

        # Resolution
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        status = solver.solve(model)
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
                "method": "CP-SAT (PPC)",
                "status": self.status,
                "solve_time": self.solve_time,
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
            "method": "CP-SAT (PPC)",
            "status": self.status,
            "solve_time": self.solve_time,
            "makespan": makespan,
            "objective": self.objective_value,
            "avg_utilization": round(avg_util, 2),
            "machine_utilization": util,
        }
