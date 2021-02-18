from flask import Flask,render_template,request
import pickle

app = Flask(__name__)

kn = pickle.load(open('Iris_kn.pkl', 'rb'))
ls = pickle.load(open('Iris_ls.pkl', 'rb'))
svc = pickle.load(open('Iris_svc.pkl', 'rb'))

@app.route('/')
def name():
    return render_template('index.html')


@app.route('/',methods=['POST'])
def predict():
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])
    data = [[sepal_length,sepal_width,petal_length,petal_width]]
    typeml = request.form['typeml']

    if typeml == "KNN":
        result = kn.predict(data)[0]
        return render_template('index.html',sepal_length=sepal_length,sepal_width=sepal_width,petal_length=petal_length,petal_width=petal_width,typeml=typeml,result=result)
    elif typeml == "LOGISTIC":
        result = ls.predict(data)[0]
        return render_template('index.html',sepal_length=sepal_length,sepal_width=sepal_width,petal_length=petal_length,petal_width=petal_width,typeml=typeml,result=result)
    elif typeml =="SVM":
        result = svc.predict(data)[0]
        return render_template('index.html',sepal_length=sepal_length,sepal_width=sepal_width,petal_length=petal_length,petal_width=petal_width,typeml=typeml,result=result)




if __name__ == '__main__':
    app.run(debug=True)