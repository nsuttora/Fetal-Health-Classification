from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

app = Flask(__name__)
model = pickle.load(open('random_forest_classification_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():    
    # feature engineering
    fh_data = pd.read_csv('fetal_health.csv')
    fh_data = fh_data.drop(columns=['histogram_width', 'histogram_min', 'histogram_max', 'histogram_number_of_peaks', 'histogram_number_of_zeroes',
                                    'histogram_mode', 'histogram_mean', 'histogram_median', 'histogram_variance', 'histogram_tendency'])
    fh_data = fh_data.rename(columns={'baseline value'                                            : 'baseline_FHR',
                                    'abnormal_short_term_variability'                           : 'abnormal_STV',
                                    'mean_value_of_short_term_variability'                      : 'mean_STV',
                                    'percentage_of_time_with_abnormal_long_term_variability'    : 'abnormal_LTV',
                                    'mean_value_of_long_term_variability'                       : 'mean_LTV'})
    fh_data['fetal_health'] = fh_data['fetal_health'].astype('uint8')

    # preproccessing
    X = fh_data.drop(columns=['fetal_health']).values
    y = fh_data['fetal_health'].values
    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.3, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # predicting on given inputs
    categories = ['Normal', 'Suspect', 'Pathological']
    if request.method == 'POST':
        Fhr = request.form['FHR']
        Acc = request.form['acc']
        Mov = request.form['movement']
        Cont = request.form['contractions']
        Ldec = request.form['ldec']
        Sdec = request.form['sdec']
        Pdec = request.form['pdec']
        PercSTV = request.form['percentSTV']
        MeanSTV = request.form['meanSTV']
        PercLTV = request.form['percentLTV']
        MeanLTV = request.form['meanLTV']
        
        pred = model.predict(sc.transform([[Fhr, Acc, Mov, Cont, Ldec, Sdec, Pdec, PercSTV, MeanSTV, PercLTV, MeanLTV]]))
        category = categories[int(pred[0])-1]
        return render_template('index.html', prediction_text='Fetal Health: {}'.format(category))
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)