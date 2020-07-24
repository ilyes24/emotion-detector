from django import forms

ALGORITHMS = [
    ('SVM', 'SVM'),
    ('KNN', 'KNN'),
    ('Naive Bayes', 'Naive Bayes'),
    ('Lexicon Dictionary', 'Lexicon Dictionary')
]


class TextInputForm(forms.Form):
    textInput = forms.CharField(widget=forms.Textarea(attrs={"rows": 8, "class": "form-control margin-left-0", "aria-describedby": "arabicTextInput", "placeholder": "Enter your text to classifie"}), label='')
    algorithm = forms.MultipleChoiceField(required=True, widget=forms.RadioSelect(attrs={"class": "custom-control-inline margin-right-0"}), choices=ALGORITHMS)
