from flask import Flask, request, render_template
from flask.ext.wtf import Form
from wtforms import StringField, SubmitField, IntegerField
from wtforms.validators import Required

from predict_boston import PredictBostonData

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

class input_data(Form):
    lstat = IntegerField()
    crim = IntegerField()
    age = IntegerField()
    submit = SubmitField('送信')


@app.route('/result_predict', methods=["GET", "POST"])
def result_predict():
    form = input_data(request.form)

    if request.method == 'POST':
        predict_boston = PredictBostonData(form.lstat.data,
                                           form.crim.data,
                                           form.age.data)
        predict = predict_boston.predict()

        return render_template('results.html',predict=predict)

    return render_template('input_data.html',
                            form=form)

if __name__ == '__main__':
    app.run()
