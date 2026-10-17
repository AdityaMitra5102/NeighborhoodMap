import sys
import os

# Activate the virtual environment site-packages
venv_path = '/var/www/html/flaskapp/venv/lib/python3.13/site-packages'
if venv_path not in sys.path:
    sys.path.insert(0, venv_path)

# Add the application directory to the system path
sys.path.insert(0, '/var/www/html/flaskapp')

from flaskapp import app as application
