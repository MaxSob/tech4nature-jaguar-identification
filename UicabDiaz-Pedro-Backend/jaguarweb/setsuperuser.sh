#!/bin/bash
# Automatin migration and set of superuser

python manage.py makemigrations

python manage.py migrate

python manage.py ensure_adminuser --user=admin --password=admin

python manage.py runserver 0.0.0.0:8000
