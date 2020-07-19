from django.shortcuts import render
import ArabicEmotionDetector.Preprocessing as pre
import ArabicEmotionDetector.NaiveBayes as nb
import ArabicEmotionDetector.SVM as svm
import ArabicEmotionDetector.KNN as knn


def index(request):
    if request.method == "GET":
        context = {}
        return render(request, 'index.html', context)

    if request.method == "POST":
        # get inputs from FORM
        text_input = request.POST['textInput']
        algorithm = request.POST['Algorithm']

        # pre-processing phase
        steamed_text = pre.clean_text(text=text_input)


        return render(request, 'index.html', {'result': 'true'})
