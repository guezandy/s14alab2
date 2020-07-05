import joblib
from flask import Flask, render_template
app = Flask(__name__)
model = joblib.load('./notebooks/regr.pkl')
linear_model = joblib.load('./notebooks/linear_model.pkl')
tree_model = joblib.load('./notebooks/tree_model.pkl')


@app.route('/')
def index():
    features = {
        'beds': 4,
        'baths': 2.5,
        'sqft': 3005,
        'age': 17,
        'lotsize': 17903.0,
        'garage': 1
    }
    values = [v for k,v in features.items()]
    params = [values] # 2D array
    one = model.predict(params)[0][0].round(1)
    two = linear_model.predict(params)[0][0].round(1)
    three = tree_model.predict(params)[0].round(1)
    print(tree_model.predict(params), three)

    return render_template('index.html', predictions={ 
        'default': {
            'prediction': one,
            'name': 'Default',
            'description': 'This model implements LinearRegression and splits train/test 80/20 ',
        },
        'linear': {
            'prediction': two,
            'name': 'Linear',
            'description': 'This model implements LinearRegression and splits train/test 66/33',
        },
        'decision_tree': {
            'prediction': three,
            'name': 'Decision Tree',
            'description': 'This model implements DecisionTreeRegressor and splits train/test 66/33',
        }
    })
