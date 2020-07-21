from django.shortcuts import render
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
            result = knn(text_input)
        elif algorithm == 'SVM':
            result = svm(text_input)
        elif algorithm == 'Naive':
            result = naive_bayes(text_input)

        return render(request, 'result.html', {'result': result})
