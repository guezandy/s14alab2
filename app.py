import joblib
from flask import Flask, render_template
app = Flask(__name__)
model = joblib.load('./notebooks/regr.pkl')
linear_model = joblib.load('./notebooks/linear_model.pkl')
tree_model = joblib.load('./notebooks/tree_model.pkl')


@app.route('/')
def index():
    one = model.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0][0].round(1)
    two = linear_model.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0][0].round(1)
    three = 'abc' # tree_model.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0][0].round(1)
    return render_template('index.html', predictions={ 
        'default': {
            'prediction': one,
            'name': 'Default',
            'description': 'This model implements LinearRegression and splits train/test 80/20 ',
        },
        'linear': {
            'prediction': two,
            'name': 'Linear',
            'description': 'This model implements LinearRegression and splits train/test 80/20',
        },
        'decision_tree': {
            'prediction': three,
            'name': 'Decision Tree',
            'description': 'This model implements DecisionTreeRegressor and splits train/test 66/33',
        }
    })
