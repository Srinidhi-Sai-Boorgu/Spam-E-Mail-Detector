from flask import Flask, render_template, request
import pickle

with open('model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/detector', methods=['POST'])
def detector():
    email_text = request.form['email_text']
    processed_email_text = [' '.join(email_text.lower().split())]
    text_vector = vectorizer.transform(processed_email_text)
    result = model.predict(text_vector)[0]
    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)