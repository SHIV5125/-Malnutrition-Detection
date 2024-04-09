from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from malApp.views import *

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', home, name="home"),
    path('about/', about, name="about"),
    path('contact/', contact, name="contact"),
    path('register/', register, name="register"),
    path('doc-register/', doc_register, name="doc_register"),
    path('signin/', signin, name="signin"),
    path('doc-signin/', doc_signin, name="doc_signin"),
    path('change-password/', change_password, name="change_password"),
    path('update-profile/', update_profile, name="update_profile"),
    path('doc-update-profile/', doc_update_profile, name="doc_update_profile"),
    path('logout-user/', logout_user, name="logout_user"),

    path('my-history/', my_history, name="my_history"),
    path('all-user/', all_user, name="all_user"),
    path('all-doctor/', all_doctor, name="all_doctor"),
    path('history-detail/<int:pid>/', history_detail, name="history_detail"),

    path('admin-signin/', admin_signin, name="admin_signin"),
    path('delete-user/<int:pid>/', delete_user, name="delete_user"),
    path('delete-doc/<int:pid>/', delete_doc, name="delete_doc"),
    path('delete-history/<int:pid>/', delete_history, name="delete_history"),
    path('predict-data/', predict_data, name="predict_data"),

    path('prediction-dashboard/<int:pid>/', prediction_dashboard, name="prediction_dashboard"),

]+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
