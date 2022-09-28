# -*- coding: utf-8 -*-
"""

"""

from flask import request, render_template, session, redirect, url_for,jsonify
from flask_app import app, api, swagger_api
import os

from views.Model_Inference_Views import Model_Inference_Views
from API.ModelInferenceAPI import ModelInferenceAPI

# Model Inference routes
app.register_blueprint(Model_Inference_Views,url_prefix="/")

# Model Inference APIs
swagger_api.add_resource(ModelInferenceAPI, '/ModelInference')


@app.route("/logoff", methods=["GET", "POST"])
def logoff():
    session.pop('username', None)
    session.pop('typeofuser', None)
    return redirect(url_for('root'))

@app.route("/error", methods=["GET", "POST"])
def error():

    error_title  = "Error Title"
    error_msg = "Error Message"
    try:
        error_title = request.get("error_title")
    except: 
        pass

    try:
        error_msg = request.get("error_msg")
    except: 
        pass

    return render_template("error.html", ERROR_TITLE=error_title,ERROR_MSG=error_msg)

@app.route("/login", methods=["POST"])
def validate_login():
    username = request.form.get('username')
    password = request.form.get('password')

    webData = {
        "login_status": False,
        "error_msg": "",
        "route": "/error?error_title=loginerror&error_msg=Login Failed",
    }

    if username=="admin" and password=="admin":
        webData["login_status"] = True
        webData["username"] = username
        webData["typeofuser"] = "admin"
        webData["default_password_changed"] = True
        webData.update({"route": url_for('ModelInference.home')})
    else:
        webData["error_msg"] = "Invalid username or password name"


    if webData["login_status"]==True:
        webData.update({"error_msg": ""})
        session['username'] = webData['username']
        session['typeofuser'] = webData["typeofuser"]
        session['maxfileuploadlimit'] = 10
        session['autoremovevalidationdata'] = "OFF"
        session['updateFreq'] = 10
    
    return jsonify(webData)

@app.route('/favicon.ico')
def favicon():
    return redirect(url_for('static', filename='images/favicon.ico'))

@app.route("/", methods=["GET","POST"])
def root():
    if "username" in session:
        return redirect(url_for('ModelInference.home'))
    else:
        return render_template("login.html")

app.config['MAX_CONTENT_PATH'] = 4096 * 4096
app.config['UPLOAD_EXTENSIONS'] = ['jpeg','.jpg', '.png']
if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0', port=5000)
    print("App is running")