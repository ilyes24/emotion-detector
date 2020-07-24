from django.http import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.urls import reverse
from .data_preparation import predict_knn, predict_svm, predict_naive_bayes, lexicon_dictionary
from .forms import TextInputForm


def index(request):
    if request.method == "GET":
        form = TextInputForm()
        context = {'form': form}
        return render(request, 'index.html', context)

    if request.method == "POST":
        # get inputs from FORM
        text_input = request.POST['textInput']
        algorithm = request.POST['algorithm']

        result = None
        accuracy = None
        if algorithm == 'KNN':
            result = predict_knn([text_input])
            accuracy = '50.89%'
        elif algorithm == 'SVM':
            result = predict_svm([text_input])
            accuracy = '81.12%'
        elif algorithm == 'Naive Bayes':
            result = predict_naive_bayes([text_input])
            accuracy = '80.67%'
        elif algorithm == 'Lexicon Dictionary':
            result = lexicon_dictionary([text_input])

        context = {'result': result, 'text_input': text_input, 'algorithm': algorithm, 'accuracy': accuracy}
        return render(request, 'result.html', context)


def result_view(request):
    return render(request, 'index.html')

