from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.urls import reverse

from .KNN import knn
from .SVM import svm
from .NaiveBayes import naive_bayes


def index(request):
    if request.method == "GET":
        context = {}
        return render(request, 'index.html', context)

    if request.method == "POST":
        # get inputs from FORM
        text_input = request.POST['textInput']
        algorithm = request.POST['Algorithm']

        result = None
        if algorithm == 'KNN':
            result = knn([text_input])
        elif algorithm == 'SVM':
            result = svm([text_input])
        elif algorithm == 'Naive':
            result = naive_bayes([text_input])

        context = {'result': result, 'text_input': text_input, 'algorithm': algorithm}
        return render(request, 'result.html', context)


def result_view(request):
    r = request
    i = 20
    return render(request, 'index.html')