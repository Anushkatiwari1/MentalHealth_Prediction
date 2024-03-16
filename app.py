from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pathlib
import textwrap

import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
GOOGLE_API_KEY = 'AIzaSyBNhQWX95vciHFS-ZsqcaY6kykgfakL3Bw'
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.0-pro-latest')

diseases = {
    0: "No Issues",
    1: "Anxiety and Depression",
    2: "Attention-Deficit/Hyperactivity Disorder (ADHD)",
    3: "Obsessive-Compulsive Disorder (OCD)",
    4: "Bipolar Disorder",
    5: "Post-traumatic stress disorder (PTSD)",
    6: "Generalized anxiety disorder (GAD)",
    7: "Borderline personality disorder (BPD)",
    8: "Panic Disorder",
}


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def gemini_call(text):
    response = gemini_model.generate_content(text)
    return response.text


app = Flask(__name__)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/form')
def form():
    return render_template('form.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final = [np.array(int_features)]
    print(int_features)
    print(final)
    predictions = model.predict(final)[0]
    print(predictions)
    # return render_template('result.html')

    if (predictions == 0):
        return render_template('result.html', result=0)

    elif (predictions == 1):
        return render_template('result.html', result=1)

    elif (predictions == 2):
        return render_template('result.html', result=2)

    elif (predictions == 3):
        return render_template('result.html', result=3)

    elif (predictions == 4):
        return render_template('result.html', result=4)

    elif (predictions == 5):
        return render_template('result.html', result=5)

    elif (predictions == 6):
        return render_template('result.html', result=6)

    elif (predictions == 7):
        return render_template('result.html', result=7)

    elif (predictions == 8):
        return render_template('result.html', result=8)


@app.route('/receive_numbers', methods=['POST'])
def receive_numbers():
    data = request.get_json()
    numbers = data.get('data')
    print(numbers)
    numbers_np = [np.array(numbers)]
    # print(numbers_np)

    for i in numbers:
        if(i == None):
            return jsonify({'prediction': 'fill all the values', 'suggestion': ''})
        else:
            predictions = model.predict(numbers_np)[0]
            print(predictions)
            if(predictions == 0):
                suggestion = gemini_call('Mention some prevention and daily practices for a good mental health')
                return jsonify({'prediction': int(predictions), 'suggestion': suggestion})
            else:

                suggestion = gemini_call(f'Mention some prevention and cures for {diseases[predictions]}')
                print(suggestion)

                return jsonify({'prediction': int(predictions), 'suggestion': suggestion})




if __name__ == "__main__":
    app.run(debug=True, port=8001)
