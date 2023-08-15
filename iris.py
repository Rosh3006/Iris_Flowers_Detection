from flask import Flask,render_template,request
import pickle

app=Flask(__name__)
model=pickle.load(open('D:\\flask\\iris classification\\saveModel.sav','rb'))

@app.route('/')
def home():
    result=''
    return render_template('irisform.html',**locals())

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='POST':
        sepal_length=float(request.form['sepal_width'])
        sepal_width = float(request.form['petal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        y = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        result=''
        if y==0:
            result="Iris setosa"
        elif y==1:
            result="Iris virginica"
        else:
            result="Iris versicolor"

    return render_template('irisform.html',**locals())

if __name__== '__main__':
    app.run(debug=True)