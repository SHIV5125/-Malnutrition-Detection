from django.contrib import admin
from .models import History, Register, Doctor

# Register your models here.
admin.site.register(Register)
admin.site.register(History)
admin.site.register(Doctor)
