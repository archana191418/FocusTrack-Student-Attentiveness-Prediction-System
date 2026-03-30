
from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)


student_data = pd.DataFrame({
    'eye_contact': [1,1,0,1,0,0,1,1,0,1],
    'mobile_usage': [0,1,1,0,1,1,0,0,1,0],
    'note_taking': [1,0,0,1,0,0,1,1,0,1],
    'attendance_percentage': [90,75,60,88,55,50,92,85,65,80],
    'speaking_in_class': [1,0,0,1,0,0,1,1,0,1],
    'yawn_count': [1,4,5,1,6,7,0,2,5,1],
    'attentive': [1,0,0,1,0,0,1,1,0,1]
})

X = student_data.drop('attentive', axis=1)
y = student_data['attentive']

model = LogisticRegression()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    eye_contact = int(request.form['eye_contact'])
    mobile_usage = int(request.form['mobile_usage'])
    note_taking = int(request.form['note_taking'])
    attendance_percentage = int(request.form['attendance_percentage'])
    speaking_in_class = int(request.form['speaking_in_class'])
    yawn_count = int(request.form['yawn_count'])

    data = [[
        eye_contact,
        mobile_usage,
        note_taking,
        attendance_percentage,
        speaking_in_class,
        yawn_count
    ]]

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = 'Student is Attentive,student is good '
    else:
        result = 'Student is not attentive' 

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)