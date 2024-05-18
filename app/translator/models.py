from django.db import models


class incentives(models.Model):
    id = models.AutoField(primary_key=True)
    seal = models.TextField()
    incentives = models.TextField()
    order_number = models.TextField()


class job_information(models.Model):
    id = models.AutoField(primary_key=True)
    date_reception = models.TextField()
    date_dismissal = models.TextField()
    sail = models.TextField()
    position = models.TextField()
    ordet = models.TextField()


class title(models.Model):
    id = models.AutoField(primary_key=True)
    series = models.TextField()
    number = models.TextField()
    last_name = models.TextField()
    first_name = models.TextField()
    middle_name = models.TextField()
    year_birth = models.TextField()
    date_filling = models.TextField()
    last_name_change = models.TextField()
    first_name_change = models.TextField()
    middle_name_change = models.TextField()