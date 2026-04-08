"""
Generateur de jeux de donnees pour l'ordonnancement sur machines paralleles.
Genere des instances avec 3 niveaux de difficulte (facile, moyen, difficile).
"""

import random
import csv
from collections import namedtuple
from datetime import datetime


taskInfo = namedtuple("taskInfo", ["duration", "predecessors", "relase_date", "due_date"])


class SchedulingDatasetGenerator:

    FACILE = {
        "num_pairs": 3,
        "num_machines": 4,
        "min_duration": 20,
        "max_duration": 80,
        "slack_factor": 0.6,
    }

    MOYEN = {
        "num_pairs": 5,
        "num_machines": 3,
        "min_duration": 30,
        "max_duration": 100,
        "slack_factor": 0.3,
    }

    DIFFICILE = {
        "num_pairs": 8,
        "num_machines": 3,
        "min_duration": 40,
        "max_duration": 120,
        "slack_factor": 0.25,
    }

    def __init__(self, seed=None):
        if seed is None:
            seed = int(datetime.now().timestamp() * 1000) % (2**32)
        random.seed(seed)
        self.current_seed = seed

    def generate_dataset(self, num_pairs=5, num_machines=3, min_duration=20,
                         max_duration=100, slack_factor=0.3, time_horizon=1000):
        """
        Genere un jeu de donnees complet.

        Returns:
            (dict de taskInfo, list de machines)
        """
        tasks = {}
        project_ids = [chr(97 + i) for i in range(num_pairs)]

        # Facteur de contention : plus il y a de taches par machine, plus il
        # faut elargir les fenetres pour rester faisable
        contention = (2 * num_pairs) / num_machines

        for pid in project_ids:
            dur_1 = random.randint(min_duration, max_duration)
            dur_2 = random.randint(min_duration, max_duration)

            release_1 = random.randint(0, time_horizon // 4)
            release_2 = release_1 + dur_1

            total_dur = dur_1 + dur_2
            # La marge tient compte du slack demande + de la contention
            slack = int(total_dur * (slack_factor + contention * 0.3))

            due_1 = min(release_1 + dur_1 + dur_2 + slack, time_horizon)
            due_2 = min(release_2 + dur_2 + slack, time_horizon)

            # task_X_1 est le predecesseur, task_X_2 est le successeur
            tasks[f"task_{pid}_1"] = taskInfo(
                duration=dur_1,
                predecessors="none",
                relase_date=release_1,
                due_date=due_1,
            )
            tasks[f"task_{pid}_2"] = taskInfo(
                duration=dur_2,
                predecessors=f"task_{pid}_1",
                relase_date=release_2,
                due_date=due_2,
            )

        machines = [f"m_{i+1}" for i in range(num_machines)]
        return tasks, machines

    def save_to_csv(self, tasks, machines, filename):
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["task_name", "duration", "predecessors", "relase_date", "due_date"])
            writer.writeheader()
            for name, info in tasks.items():
                writer.writerow({
                    "task_name": name,
                    "duration": info.duration,
                    "predecessors": info.predecessors,
                    "relase_date": info.relase_date,
                    "due_date": info.due_date,
                })
            writer.writerow({})
            writer.writerow({
                "task_name": "MACHINES",
                "duration": ",".join(machines),
                "predecessors": "",
                "relase_date": "",
                "due_date": "",
            })

    def load_from_csv(self, filename):
        tasks = {}
        machines = []
        with open(filename, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
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
        return tasks, machines


def main():
    gen = SchedulingDatasetGenerator(seed=42)

    for level_name, config in [("facile", gen.FACILE), ("moyen", gen.MOYEN), ("difficile", gen.DIFFICILE)]:
        tasks, machines = gen.generate_dataset(**config)
        gen.save_to_csv(tasks, machines, f"dataset_{level_name}.csv")
        print(f"dataset_{level_name}.csv genere : {len(tasks)} taches, {len(machines)} machines")


if __name__ == "__main__":
    main()
