from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)

with open("gb_model.pickle",'rb') as f:
    gb_model = pickle.load(f)
f.close()

with open("encoder.pickle",'rb') as f:
    encoder = pickle.load(f)
f.close()

with open("minmax.pickle",'rb') as f:
    minmax = pickle.load(f)
f.close()

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods=['POST'])
def predict():
    
    gender=int(request.form.get("gender"))
    age=int(request.form.get("age"))
    region_code=int(request.form.get("region_code"))
    occupation=str(request.form.get("occupation"))
    channel_code=str(request.form.get("channel_code"))    
    vintage=int(request.form.get("vintage"))    
    credit_product=int(request.form.get("credit_product")) 
    avg_acc_bal=np.log(int(request.form.get("avg_account_balance")))    
    is_active=int(request.form.get("is_active"))              
    
    numericals = np.array([gender, age, region_code, vintage, credit_product, avg_acc_bal, is_active])
    encoded = np.array(encoder.transform(np.array([occupation, channel_code]).reshape(1,-1)).todense()).squeeze()
    numericals[1:4] = minmax.transform(numericals[1:4].reshape(1,-1)).squeeze()
    input_x = np.hstack((numericals, encoded)).reshape(1,-1)
    output = gb_model.predict(input_x)[0]

    if output==1:
        result='is a potential lead'
    else:
        result='is not a potential lead'
        
    return render_template('result.html',  prediction_text="This customer {}".format(result)) 

if __name__=='__main__':
    app.run(port=8000,debug=True) 
    
