"""jaguarweb URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
#from django.contrib import admin
from django.urls import path
from dashboard import views
from django.contrib.auth.views import LoginView,LogoutView
from django.contrib import admin


urlpatterns = [
    path("admin/", admin.site.urls, name='adminview'),
    path('',views.home,name='home'),
    path('dashboard/',views.dashboard,name='dashboard'),
    path('demo/',views.demoModels,name='demo'),
    #path('register/',views.register,name='register'),
    path('login/',LoginView.as_view(template_name='account/login.html'),name='login'),
    path('logout/',LogoutView.as_view(template_name='home.html'),name='logout'),
    path('resetpassword/',views.resetPass,name='resetpassword'),
    path('demo/objectdet/',views.objectDetection,name='objectdet'), # OBJECT DETECTION
    path('demo/objectdet/objdect',views.detection,name='detec') # OBJECT DETECTION
]
