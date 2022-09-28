"""
"""
from flask import Blueprint
from flask import request, render_template, session, redirect, url_for,jsonify
import json

Model_Inference_Views = Blueprint('ModelInference', __name__, 
    template_folder='templates', 
    static_folder='static', 
    static_url_path='assets')

import datetime
# Helper functions starts

def parseArg(req_instance, key):
    Param = None
    try:
        Param = req_instance.form.get(key)
    except Exception as e:
        print(f"error getting param {key} error: {e}")
    return Param

def return_with_errorMsg(error_msg):
    response_data = {"error": True, "error_msg": error_msg}
    return response_data

# Helper functions End

@Model_Inference_Views.route("/home", methods=["GET", "POST"])
def home():
    if "username" in session:
        return render_template("ModelInference/home.html", username=session["username"], role=session['typeofuser'])
    else:
        return redirect(url_for('root'))