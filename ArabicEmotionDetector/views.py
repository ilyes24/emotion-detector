from django.shortcuts import render


def index(request):
    if request.method == "GET":
        context = {}
        return render(request, 'index.html', context)

    if request.method == "POST":
        text = request.POST['textInput']
        algorithm = request.POST['Algorithm']

        return render(request, 'index.html', {'result': 'true'})


