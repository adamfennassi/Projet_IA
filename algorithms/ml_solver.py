"""
Solveur ML pur pour l'ordonnancement sur machines paralleles.
Reseau de neurones PyTorch entraine sur des solutions CP-SAT,
puis utilise en inference pour construire un ordonnancement via dispatching.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .cp_sat_solver import CPSatSolver


class SchedulingNet(nn.Module):
    """
    Reseau de neurones a 2 tetes :
    - Classification : predit la machine pour chaque tache
    - Regression : predit la priorite (ordre de dispatching)
    """

    def __init__(self, input_dim, num_machines, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
        )
        self.machine_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_machines),
        )
        self.priority_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        h = self.shared(x)
        return self.machine_head(h), self.priority_head(h).squeeze(-1)


def _extract_features(tasks, machines):
    """Extrait 7 features normalisees par tache."""
    max_time = max(max(t["due_date"] for t in tasks.values()), 1)
    max_dur = max(t["duration"] for t in tasks.values())
    n_tasks = len(tasks)
    n_machines = len(machines)

    features = {}
    for name, info in tasks.items():
        slack = info["due_date"] - info["relase_date"] - info["duration"]
        pred_dur = 0
        pred = info["predecessors"]
        if pred != "none" and pred in tasks:
            pred_dur = tasks[pred]["duration"]

        features[name] = [
            info["duration"] / max_dur,
            info["relase_date"] / max_time,
            info["due_date"] / max_time,
            slack / max_time,
            1.0 if pred != "none" else 0.0,
            pred_dur / max_dur if max_dur > 0 else 0.0,
            n_tasks / n_machines,
        ]
    return features


INPUT_DIM = 7  # nombre de features par tache
MAX_MACHINES = 10  # nombre max de machines supportees par le modele


class MLSolver:
    """
    Solveur ML pur :
    1. Entraine un reseau sur des solutions CP-SAT
    2. Predit affectation machine + priorite
    3. Construit un ordonnancement par dispatching glouton
    """

    def __init__(self, tasks, machines, model=None):
        self.tasks = tasks
        self.machines = machines
        self.num_machines = len(machines)
        self.model = model
        self.solution = None
        self.solve_time = None
        self.status = None
        self.objective_value = None
        self.training_time = None

    def _generate_training_data(self, generator, num_instances=200):
        """Genere X, y_machine, y_priority en resolvant num_instances avec CP-SAT."""
        all_X, all_ym, all_yp = [], [], []

        configs = []
        per = num_instances // 3
        configs += [generator.FACILE] * per
        configs += [generator.MOYEN] * per
        configs += [generator.DIFFICILE] * (num_instances - 2 * per)

        for config in configs:
            tasks_nt, machines = generator.generate_dataset(**config)
            tasks_dict = {
                n: {"duration": t.duration, "predecessors": t.predecessors,
                    "relase_date": t.relase_date, "due_date": t.due_date}
                for n, t in tasks_nt.items()
            }

            solver = CPSatSolver(tasks_dict, machines, time_limit=10)
            sol = solver.solve()
            if sol is None:
                continue

            features = _extract_features(tasks_dict, machines)
            m2i = {m: i for i, m in enumerate(machines)}
            max_start = max(max(s["start"] for s in sol.values()), 1)

            for name in tasks_dict:
                all_X.append(features[name])
                all_ym.append(m2i[sol[name]["machine"]])
                all_yp.append(sol[name]["start"] / max_start)

        if not all_X:
            return None, None, None

        return (
            torch.tensor(all_X, dtype=torch.float32),
            torch.tensor(all_ym, dtype=torch.long),
            torch.tensor(all_yp, dtype=torch.float32),
        )

    def train(self, generator, num_instances=200, epochs=100, lr=0.001, batch_size=64):
        """
        Entraine le modele sur des solutions CP-SAT.
        Returns: dict {total_loss, machine_acc} par epoque
        """
        t0 = time.time()

        X, y_m, y_p = self._generate_training_data(generator, num_instances)
        if X is None:
            self.training_time = time.time() - t0
            return {"error": "Pas assez de donnees d'entrainement"}

        self.model = SchedulingNet(INPUT_DIM, MAX_MACHINES)
        loader = DataLoader(TensorDataset(X, y_m, y_p), batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        ce = nn.CrossEntropyLoss()
        mse = nn.MSELoss()

        history = {"total_loss": [], "machine_acc": []}

        for epoch in range(epochs):
            self.model.train()
            total_loss, correct, total = 0, 0, 0

            for bx, bm, bp in loader:
                optimizer.zero_grad()
                logits, prio = self.model(bx)
                loss = ce(logits, bm) + 0.5 * mse(prio, bp)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * bx.size(0)
                correct += (logits.argmax(1) == bm).sum().item()
                total += bx.size(0)

            history["total_loss"].append(total_loss / total)
            history["machine_acc"].append(correct / total)

        self.training_time = time.time() - t0
        return history

    def solve(self):
        """
        Construit un ordonnancement par dispatching guide par le ML.
        Returns: dict solution ou None
        """
        if self.model is None:
            self.status = "NOT_TRAINED"
            return None

        t0 = time.time()
        self.model.eval()

        features = _extract_features(self.tasks, self.machines)
        names = list(self.tasks.keys())
        X = torch.tensor([features[n] for n in names], dtype=torch.float32)

        with torch.no_grad():
            logits, priorities = self.model(X)

        machine_preds = logits.argmax(1).numpy()
        prios = priorities.numpy()

        # Trier par priorite (les plus petites = plus urgentes)
        order = sorted(range(len(names)), key=lambda i: prios[i])

        machine_end = {m: 0 for m in self.machines}
        task_ends = {}
        solution = {}

        for idx in order:
            name = names[idx]
            info = self.tasks[name]

            # Plus tot possible (release date + fin du predecesseur)
            earliest = info["relase_date"]
            pred = info["predecessors"]
            if pred != "none" and pred in task_ends:
                earliest = max(earliest, task_ends[pred])

            # Machine predite
            pred_m_idx = machine_preds[idx]
            preferred = self.machines[pred_m_idx] if pred_m_idx < len(self.machines) else self.machines[0]

            # Choisir la machine avec le meilleur demarrage
            best_m = preferred
            best_start = max(earliest, machine_end[preferred])

            for m in self.machines:
                cand = max(earliest, machine_end[m])
                if cand < best_start:
                    best_start = cand
                    best_m = m

            end = best_start + info["duration"]

            # Verifier due_date, sinon essayer une autre machine
            if end > info["due_date"]:
                for m in self.machines:
                    cand = max(earliest, machine_end[m])
                    if cand + info["duration"] <= info["due_date"]:
                        best_m, best_start = m, cand
                        end = best_start + info["duration"]
                        break

            solution[name] = {
                "start": best_start,
                "end": end,
                "duration": info["duration"],
                "machine": best_m,
            }
            machine_end[best_m] = end
            task_ends[name] = end

        self.solve_time = time.time() - t0
        self.solution = solution
        self.objective_value = sum(s["start"] for s in solution.values())

        # Verifier faisabilite
        feasible = True
        for name, info in self.tasks.items():
            s = solution[name]
            if s["end"] > info["due_date"]:
                feasible = False
                break
            pred = info["predecessors"]
            if pred != "none" and pred in solution:
                if solution[pred]["end"] > s["start"]:
                    feasible = False
                    break

        self.status = "FEASIBLE" if feasible else "INFEASIBLE_APPROX"
        return self.solution

    def get_makespan(self):
        if not self.solution:
            return None
        return max(v["end"] for v in self.solution.values())

    def get_kpis(self):
        if not self.solution:
            return {
                "method": "ML (PyTorch)",
                "status": self.status or "NO_SOLUTION",
                "solve_time": self.solve_time,
                "training_time": self.training_time,
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
            "method": "ML (PyTorch)",
            "status": self.status,
            "solve_time": self.solve_time,
            "training_time": self.training_time,
            "makespan": makespan,
            "objective": self.objective_value,
            "avg_utilization": round(avg_util, 2),
            "machine_utilization": util,
        }
