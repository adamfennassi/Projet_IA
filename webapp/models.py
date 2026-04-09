from django.db import models


class BenchmarkResult(models.Model):
    name = models.CharField(max_length=200, default="Benchmark")
    created_at = models.DateTimeField(auto_now_add=True)
    results_json = models.TextField()
    dashboard_image = models.ImageField(upload_to="dashboards/", blank=True)
    gantt_image = models.ImageField(upload_to="gantts/", blank=True)

    def __str__(self):
        return f"{self.name} - {self.created_at.strftime('%d/%m/%Y %H:%M')}"

    class Meta:
        ordering = ["-created_at"]