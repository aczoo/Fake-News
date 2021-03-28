from django.urls import path

from .views import *

urlpatterns = [
    path('', MyView),
    path('result', MyResultView),

]