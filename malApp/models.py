from re import T
from django.db import models
from  django.contrib.auth.models import User

# Create your models here.
class Register(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    mobile = models.CharField(max_length=10, null=True, blank=True)
    dob = models.DateField(null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    image = models.FileField(null=True, blank=True)

    def __str__(self) -> str:
        return self.user.username

class Doctor(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    mobile = models.CharField(max_length=10, null=True, blank=True)
    dob = models.DateField(null=True, blank=True)
    specialization = models.CharField(max_length=30, null=True, blank=True)
    experience = models.CharField(max_length=30, null=True, blank=True)
    address = models.TextField(null=True, blank=True)
    image = models.FileField(null=True, blank=True)

    def __str__(self) -> str:
        return self.user.username

class History(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    img = models.FileField(null=True, blank=True)
    preds = models.CharField(max_length=30, null=True, blank=True)
    max_prob = models.CharField(max_length=30, null=True, blank=True)
    class_name = models.CharField(max_length=30, null=True, blank=True)
    gender = models.CharField(max_length=30, null=True, blank=True)
    age = models.CharField(max_length=30, null=True, blank=True)
    weight = models.CharField(max_length=30, null=True, blank=True)
    height = models.CharField(max_length=30, null=True, blank=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.user.username