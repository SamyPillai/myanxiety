
import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
model = joblib.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = list()

    gender = ['Male','Female','Transgender','Other']
    marital = ['Single','Married','Divorced','Widower','Separated']
    family = ['Alone','Nuclear','Joint','Extended Family']
    working = ['Healthcare worker currently dealing with patient',
                'Working (coming across lot of different peoples)',
                'Working from home','Not working']
    hereditary = ['Yes','No']
    substance = ['alcohol','cigarette','narcotics','None']

    gender_le = LabelEncoder()
    marital_le = LabelEncoder()
    family_le = LabelEncoder()
    working_le = LabelEncoder()
    hereditary_le = LabelEncoder()
    substance_le = LabelEncoder()

    gender_code = gender_le.fit_transform(gender)
    marital_code = marital_le.fit_transform(marital)
    family_code = family_le.fit_transform(family)
    working_code = working_le.fit_transform(working)
    hereditary_code = hereditary_le.fit_transform(hereditary)
    substance_code = substance_le.fit_transform(substance)

    print("The genders are ")
    print(gender_code)
    print(gender_le.inverse_transform(gender_code))

    age = request.form.get('Age')
    gen = request.form.get('gender')
    mar = request.form.get('marital')
    fam = request.form.get('family')
    work = request.form.get('working')
    her = request.form.get('hereditary')
    subs = request.form.get('substance')

    gen_ind = gender.index(gen)  
    gender_ip = gender_code[gen_ind]

    mar_ind = marital.index(mar)  
    marital_ip = marital_code[mar_ind]

    fam_ind = family.index(fam)  
    family_ip = family_code[fam_ind]

    work_ind = working.index(work)  
    working_ip = working_code[work_ind]      

    her_ind = hereditary.index(her)  
    hereditary_ip = hereditary_code[her_ind]

    subs_ind = substance.index(subs)  
    substance_ip = substance_code[subs_ind]    

    input_features.append(age)
    input_features.append(gender_ip)
    input_features.append(marital_ip)
    input_features.append(family_ip)
    input_features.append(working_ip)
    input_features.append(hereditary_ip)
    input_features.append(substance_ip)

    final_features = np.array(input_features)

    final_features = np.reshape(final_features,(1, final_features.size))

    prediction = model.predict(final_features)

    if prediction == 0:
        output = 'Mild/Moderate'
    elif prediction == 1:
        output = 'Normal'

    return render_template('index.html', prediction_text='Anxiety Level : {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    #data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])

    #output = prediction[0]
    #return jsonify(output)
    return "hi"

if __name__ == "__main__":
    app.run(debug=True)