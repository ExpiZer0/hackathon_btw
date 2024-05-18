from django.shortcuts import render
from django.views.generic.edit import FormView
from .forms import FileFieldForm
from django.http import HttpResponse
from django.template import loader


class FileFieldFormView(FormView):
    form_class = FileFieldForm()
    template_name = "translator/index.html"
    success_url = ""

    def form_valid(self, form):
        files = form.cleaned_data["file_field"]
        for f in files:
            # TODO: Do something with each file.
            ...
        return super().form_valid(form)


def index(request):
    if request.method == "POST":
        form = FileFieldForm(request.POST)
    else:
        form = FileFieldForm()
    return render(
        request,
        "translator/index.html",
        {
            "form": form
        })
