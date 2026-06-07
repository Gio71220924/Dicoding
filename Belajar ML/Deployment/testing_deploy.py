from flask import Flask, request, jsonify
import joblib

#Inisialisasi Flask app
app = Flask(__name__)

# Memuat model yang telah disimpan
joblib_model = joblib.load('gbr_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()['data'] #Mengambil data dari request JSON
    prediction = joblib_model.predict(data) #Melakukan prediksi menggunakan model 
    return jsonify({'prediction': prediction.tolist()}) #Mengembalikan hasil prediksi dalam format JSON

if __name__ == '__main__':
    app.run(debug=True) #Menjalankan Flask app dengan mode debug