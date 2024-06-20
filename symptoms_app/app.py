from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = Flask(__name__)

def neural_network(parameters):
    parameters_2d = tf.constant([parameters], dtype=tf.float32)
    model = Sequential()
    model.add(Dense(1, input_dim=9, activation='sigmoid'))
    # model.add(Dense(8, activation='relu'))
    # model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    custom_weights = [
        [0.3],
        [0.25],
        [0.005],
        [0.009],
        [0.09],
        [0.001],
        [0.0001],
        [0.001],
        [0.005],
    ]
    
    custom_biases = [0.0] * 1
    
    model.layers[0].set_weights([tf.constant(custom_weights), tf.constant(custom_biases)])
    
    prediction = model.predict(parameters_2d)

    output = 'You should take an MRI' if prediction > 0.8 else 'Consult a doctor before taking an MRI' 
    
    return output
    

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        # Get parameter values from the form
        parameter1 = int(request.form['parameter1'])
        parameter2 = int(request.form['parameter2'])
        parameter3 = int(request.form['parameter3'])
        parameter4 = int(request.form['parameter4'])
        parameter5 = int(request.form['parameter5'])
        parameter6 = int(request.form['parameter6'])
        parameter7 = int(request.form['parameter7'])
        parameter8 = int(request.form['parameter8'])
        parameter9 = int(request.form['parameter9'])

        # Do something with the parameters (e.g., print them)
        print("Parameter 1:", parameter1)
        print("Parameter 2:", parameter2)
        print("Parameter 3:", parameter3)
        print("Parameter 4:", parameter4)
        print("Parameter 5:", parameter5)
        print("Parameter 6:", parameter6)
        print("Parameter 7:", parameter7)
        print("Parameter 8:", parameter8)
        print("Parameter 9:", parameter9)

        parameters = [parameter1, parameter2, parameter3, parameter4, parameter5, parameter6, parameter7, parameter8, parameter9]
        
        result=neural_network(parameters)
        
        print(parameters)
        print(result)
        
        # You can perform further processing or redirect to another page
        
    return render_template('index.html',result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
