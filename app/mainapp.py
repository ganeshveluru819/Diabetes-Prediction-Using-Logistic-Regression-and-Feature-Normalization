
import numpy as np
from flask import Flask, render_template,request


# App config.

app = Flask(__name__)




#Define home route
@app.route("/")
def home():
    return render_template("index.html")

#Define diagnosis route
@app.route("/detection", methods=['POST'])
def detection():
    person = request.form['person']
    age = int(request.form['age'])
    pregnant = int(request.form['preg'])
    ins =int( request.form['insulin'])
    st=int( request.form['skinthick'])
    bmi = float(request.form['bmi'])
    pedi = float(request.form['pedigree'])
    glucose = int(request.form['glucose'])
    bp = int(request.form['bp'])
    a=np.array([[0.31703961], [0.85131886], [0.05494208], [0.47986301], [0.6643394],  [0.26721049],[0.28365182], [0.20666754]])
    #X=np.array([[6.0, 162.0, 62.0, 33.0, 206.8, 24.3, 0.17800000000000002, 50.0]])
    X=np.array([[pregnant,glucose,bp,st,ins,bmi,pedi,age]])
    mean=np.array([3.84505208,121.69479167,72.38932292 ,29.20442708,156.96822917,31.99257812,0.4718763,33.24088542])
    std=np.array([3.36738361,30.44307442,12.09815501 ,8.92816419,88.80652092 ,7.87902573,0.33111282,11.75257265])
    
    X_norm=(X-mean)/std
    bi=-0.3303646624320235
    res=np.dot(X_norm,a)+bi
    mp=1.0 / (1 + np.exp(-res))
    if mp>=0.5:
        predicted_class=1
    else:
        predicted_class=0
    
   
    
    if int(predicted_class) == 1:
        return render_template("positive.html", result="true",name=person)
    elif int(predicted_class) == 0:
        return render_template("negetive.html", result="true")
        
if __name__ == "__main__":
    app.run(DEBUG = True)