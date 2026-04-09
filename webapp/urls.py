from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_csv, name='upload_csv'),
    path('benchmark/', views.run_benchmark_view, name='benchmark'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('gantt/<int:result_id>/', views.gantt_view, name='gantt'),
    path('delete/<int:result_id>/', views.delete_result, name='delete_result'),
    path('delete-all/', views.delete_all, name='delete_all'),
]