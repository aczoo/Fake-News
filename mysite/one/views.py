from django.http import HttpResponse
from django.shortcuts import render
from django.views import View
from .forms import *
from .predit import *
def MyView(request):
    if request.method == 'POST':
        form = TextForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            tester = myModel()
            tester.model()
            x=form.cleaned_data["text"]
            y=tester.predict([x])
            return render(request, 'results.html', {'text':x, 'result':y})

    else:
        return render(request, 'base.html')
def MyResultView(request):
    return render(request, 'results.html', {'text':"hi", 'result':"bye"})        
