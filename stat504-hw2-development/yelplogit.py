import numpy as np
#need to 'conda install flask' for this to work
from flask import Flask, abort, jsonify, request
import cPickle as pickle


yelp_logreg = pickle.load(open("yelplogit.pkl", "rb"))

app = Flask(__name__)

@app.route('/api', methods=['POST'])
def make_predict():
    #all kinds of error checking should go here
    data = request.get_json(force=True)
    #convert our json to a numpy array
    predict_request = [data['cool'],data['useful'],data['funny']] 
    predict_request = np.array(predict_request)
    #np array goes into random forest, prediction comes out
    y_hat = yelp_logreg.predict(predict_request)
    #return our prediction
    output = [y_hat[0]]
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(host='129.101.160.58', port = 5000, debug = True)
