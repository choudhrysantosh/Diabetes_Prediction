from flask import Flask,render_template,request
import pickle



app = Flask(__name__)
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    pregnencies = int(request.form['pregnencies'])
    glucose = float(request.form['glucose'])
    blood_presure = float(request.form['blood_presure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    BMI = float(request.form['BMI'])
    diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
    age = int(request.form['age'])

    model = pickle.load(open(r'modelForPrediction.pkl', 'rb'))
    scaler = pickle.load(open(r'sandardScalar.pkl', 'rb'))
    prediction = model.predict(scaler.transform([[pregnencies,glucose,blood_presure,skin_thickness,insulin,BMI,diabetes_pedigree_function,age]]))
    output = prediction[0]
    print(output)
    if output == 1:
        return render_template('index.html', prediction_text="PATIENT HAS DIABETES")
    else:
        return render_template('index.html', prediction_text="PATIENT IS HEALTHY")

if __name__=="__main__":
    app.run(debug=True)