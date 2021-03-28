from django.http import HttpResponse
from django.shortcuts import render
from django.views import View
from sklearn.feature_extraction.text import TfidfVectorizer
from .forms import *
from .predit import *
convert = ["REAL NEWS","FAKE NEWS"]
def MyView(request):
    if request.method == 'POST':
        form = TextForm(request.POST)
        # check whether it's valid:
        if form.is_valid():
            loaded_model = pickle.load(open("./one/static/model.sav", 'rb'))
            vectorizer= pickle.load(open("./one/static/vectorizer.sav", 'rb'))
            text=form.cleaned_data["text"]
            x=vectorizer.transform([text])
            y=convert[loaded_model.predict(x)[0]]
            return render(request, 'results.html', {'text':text, 'result':y})

    else:
        return render(request, 'base.html')
def MyResultView(request):
    return render(request, 'results.html', {'text':"hi", 'result':"bye"})        
