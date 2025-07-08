from flask import Flask, request, render_template
import joblib
import os
import fitz  # PyMuPDF
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load TF-IDF vectorizer and model
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("model.pkl")

# Label mapping
label_mapping = {
    0: 'Advocate', 1: 'Arts', 2: 'Automation Testing', 3: 'Blockchain',
    4: 'Business Analyst', 5: 'Civil Engineer', 6: 'Data Science', 7: 'Database',
    8: 'DevOps Engineer', 9: 'DotNet Developer', 10: 'ETL Developer',
    11: 'Electrical Engineering', 12: 'Health and fitness', 13: 'HR', 14: 'Hadoop',
    15: 'Java Developer', 16: 'Mechanical Engineer', 17: 'Network Security Engineer',
    18: 'Operations Manager', 19: 'PMO', 20: 'Python Developer', 21: 'Sales',
    22: 'SAP Developer', 23: 'Testing', 24: 'Web Designing'
}

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['resume']
    if uploaded_file and uploaded_file.filename.endswith('.pdf'):
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(filepath)

        # Extract text from PDF
        resume_text = extract_text_from_pdf(filepath)

        # Predict
        vector = tfidf.transform([resume_text])
        prediction = model.predict(vector)[0]
        predicted_category = label_mapping.get(prediction, "Unknown")

        probas = model.predict_proba(vector)[0]
        confidence = round(probas[prediction] * 100, 2)

        return render_template('index.html',
                               prediction=predicted_category,
                               confidence=confidence,
                               resume_text=resume_text[:1000])  # limit preview

    else:
        return "Please upload a valid PDF file."

if __name__ == '__main__':
    app.run(debug=True)
