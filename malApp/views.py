import json
from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .models import *
from .utils import *
from django.conf import settings
from operator import itemgetter
import os
import csv
from .predict import *
import pandas as pd
import pickle as pkl
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Create your views here.
def home(request):
    return render(request, "home.html", locals())

def about(request):
    return render(request, "about.html", locals())

def contact(request):
    return render(request, "contact.html", locals())

def register(request):
    if request.method == "POST":
        re = request.POST
        rf = request.FILES
        user = User.objects.create_user(username=re['username'], first_name=re['first_name'], last_name=re['last_name'], password=re['password'])
        register = Register.objects.create(user=user, address=re['address'], mobile=re['mobile'], image=rf['image'])
        messages.success(request, "Registration Successful")
        return redirect('signin')
    return render(request, "signup.html", locals())

def doc_register(request):
    if request.method == "POST":
        re = request.POST
        rf = request.FILES
        user = User.objects.create_user(username=re['username'], first_name=re['first_name'], last_name=re['last_name'], password=re['password'])
        register = Doctor.objects.create(user=user, address=re['address'], mobile=re['mobile'], image=rf['image'], specialization=re['specialization'], experience=re['experience'])
        messages.success(request, "Registration Successful")
        return redirect('doc_signin')
    return render(request, "doc-signup.html", locals())

def update_profile(request):
    register = Register.objects.get(user=request.user)
    if request.method == "POST":
        re = request.POST
        rf = request.FILES
        try:
            image = rf['image']
            data = Register.objects.get(user=request.user)
            data.image = image
            data.save()
        except:
            pass
        user = User.objects.filter(id=request.user.id).update(username=re['username'], first_name=re['first_name'], last_name=re['last_name'])
        register = Register.objects.filter(user=request.user).update(address=re['address'], mobile=re['mobile'])
        register  = Register.objects.get(user=request.user)
        messages.success(request, "Updation Successful")
        return redirect('update_profile')
    return render(request, "update_profile.html", locals())

def doc_update_profile(request):
    doctor = Doctor.objects.get(user=request.user)
    if request.method == "POST":
        re = request.POST
        rf = request.FILES
        try:
            image = rf['image']
            data = Doctor.objects.get(user=request.user)
            data.image = image
            data.save()
        except:
            pass
        user = User.objects.filter(id=request.user.id).update(username=re['username'], first_name=re['first_name'], last_name=re['last_name'])
        doctor = Doctor.objects.filter(user=request.user).update(address=re['address'], mobile=re['mobile'], specialization=re['specialization'], experience=re['experience'])
        messages.success(request, "Updation Successful")
        return redirect('doc_update_profile')
    return render(request, "doc_update_profile.html", locals())

def signin(request):
    if request.method == "POST":
        re = request.POST
        user = authenticate(username=re['username'], password=re['password'])
        if user:
            login(request, user)
            messages.success(request, "Logged in successful")
            return redirect('home')
    return render(request, "signin.html", locals())

def doc_signin(request):
    if request.method == "POST":
        re = request.POST
        user = authenticate(username=re['username'], password=re['password'])
        if user:
            login(request, user)
            messages.success(request, "Logged in successful")
            return redirect('home')
    return render(request, "doc-signin.html", locals())
    
def admin_signin(request):
    if request.method == "POST":
        re = request.POST
        user = authenticate(username=re['username'], password=re['password'])
        if user.is_staff:
            login(request, user)
            messages.success(request, "Logged in successful")
            return redirect('home')
    return render(request, "admin_signin.html", locals())


def change_password(request):
    if request.method == "POST":
        re = request.POST
        user = authenticate(username=request.user.username, password=re['old-password'])
        if user:
            if re['new-password'] == re['confirm-password']:
                user.set_password(re['confirm-password'])
                user.save()
                messages.success(request, "Password changed successfully")
                return redirect('home')
            else:
                messages.success(request, "Password mismatch")
        else:
            messages.success(request, "Wrong password")
    return render(request, "change_password.html", locals())


def logout_user(request):
    logout(request)
    messages.success(request, "Logout Successfully")
    return redirect('home')


def my_history(request):
    history = None
    try:
        data_user = Register.objects.get(user=request.user)
        history = History.objects.filter(user=request.user)
    except:
        try:
            data_user = Doctor.objects.get(user=request.user)
            nor_user = Register.objects.filter(address=data_user.address)
            normal_user_id = [i.user.id for i in nor_user]
            history = History.objects.filter(user__id__in=normal_user_id)
        except:
            pass
    if request.user.is_staff:
        history = History.objects.filter()
    return render(request, "my_history.html", locals())

def all_user(request):
    data = Register.objects.filter()
    return render(request, "all_user.html", locals())

def all_doctor(request):
    data = Doctor.objects.filter()
    return render(request, "all_doctor.html", locals())

def history_detail(request, pid):
    history = History.objects.get(id=pid)
    product = (history.product).replace("'", '"')
    product = json.loads(str(product))
    product = product['object']
    product = sorted(product, key=itemgetter('price'))
    try:
        user = Register.objects.get(user=history.user)
    except:
        pass
    return render(request, "history_detail.html", locals())


def delete_user(request, pid):
    user = User.objects.get(id=pid)
    user.delete()
    messages.success(request, "User Deleted")
    return redirect('all_user')

def delete_doc(request, pid):
    user = User.objects.get(id=pid)
    user.delete()
    messages.success(request, "Doctor Deleted")
    return redirect('all_doctor')


def delete_history(request, pid):
    data = History.objects.get(id=pid)
    data.delete()
    messages.success(request, "History Deleted")
    return redirect('my_history')


def predict_data(request):
    datapath = str(settings.BASE_DIR) + "/MalnutritionApp/"
    mediaPath = str(settings.BASE_DIR) + "/media/"
    output = None
    if request.method == "POST":
        re = request.POST
        gender = re['gender']
        age = re['age']
        weight = re['weight']
        height = re['height']
        image_file = request.FILES['img']
        history_obj = History.objects.create(user=request.user, img=image_file, age=age, weight=weight, height=height, gender=gender)
        img_path = mediaPath + str(history_obj.img)
        model_path = datapath + "malnutrition.h5"
        img, preds, max_prob, class_name = pred_result(img_path, model_path)
        print("prediction:", preds)
        print("highest probability:", max_prob)
        if class_name == "healthy_nails":
            output = "The uploaded image belongs to "+class_name+"class." + "and the person don't have to worry. Hence He has a healthy nails."
            print("The uploaded image belongs to ", class_name, "class.",
                  "and the person don't have to worry. Hence He has a healthy nails")

        elif class_name == "healthy_skin":
            output = "The uploaded image belongs to "+class_name+"class." + "and the person don't have to worry. Hence He has a healthy skin."
            print("The uploaded image belongs to ", class_name, "class.",
                  "and the person don't have to worry. Hence He has a healthy skin")

        elif class_name == "unhealthy_nails":
            output = "The uploaded image belongs to "+class_name+"class."+"and the person have to clean theirs nails. Hence Report is Negative."
            print("The uploaded image belongs to ", class_name, "class.",
                  "and the person have to clean theirs nails. Hence Report is Negative")

        elif class_name == "unhealthy_skin":
            output = "The uploaded image belongs to "+class_name+"class."+"and the person need to visit the doctor. Hence Report is positive."
            print("The uploaded image belongs to ", class_name, "class.",
                  "and the person need to visit the doctor. Hence Report is positive")
        
        his_data = History.objects.filter(id=history_obj.id).update(preds=preds, max_prob=max_prob, class_name=class_name)
        messages.success(request, "Prediction Saved")
        return redirect('prediction_dashboard', history_obj.id)
    return render(request, "predict_data.html", locals())

def prediction_dashboard(request, pid):
    data = History.objects.get(id=pid)
    reg = Register.objects.get(user=data.user)
    doctor = Doctor.objects.filter(address=reg.address)
    return render(request, 'predicton_dashboard.html', locals())