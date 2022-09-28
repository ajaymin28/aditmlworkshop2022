#!/bin/sh
gunicorn -b 0.0.0.0:5000 --workers 1 --threads 50 --timeout 3000 app:app
# gunicorn --bind=0.0.0.0:5000 --workers=4 app:app