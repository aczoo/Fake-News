from django.http import HttpResponse
from django.shortcuts import render
from django.views import View

def MyView(request):
    return render(request, 'base.html',{})

