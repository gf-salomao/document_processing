from django.urls import path
from .views import extract_entities_view

urlpatterns = [
    path("extract_entities/", extract_entities_view),
]
