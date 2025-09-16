# %%
"""%pip install streamlit
%pip install PyMuPDF
%pip install docx2txt"""

# %%
#import streamlit as st
import re
import nltk
import fitz 
import docx2txt
import joblib
import json




# %%
#%pip install streamlit

# %%
import streamlit as st

# %%
import nltk
nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
sw = set(stopwords.words("english"))
nltk.download("wordnet")


# %%
def preprocess(resume):
    resume = re.sub("[^a-zA-Z]"," ",resume)
    text = resume.lower()
    text = text.split()
    #tokens = word_tokenize(resume)
    text = [words for words in text if words not in sw]
    #text = [lemmatizer.lemmatize(word) for word in text]
    #text = [stemmer.stem(word)for word in text]
    text =" ".join(text)
    return text

# %%
def extract_file(resume):
    fn = resume.name,lower()
    with open(fn, "wb")as f:
        f.write(resume.read())

    if fn.endwith(".pdf"):
        try:
            pdf_file = fitz.open(fn)
            text = ""
            for page in pdf_file:
                text += page.get_text()
            return text
        except Exception as e:
            return f"error reading the file {e}"
   
    elif fn.endwith(".docx"):
        try:
            docxfile = docx2txt.process(fn)
            return docxfile
        except Exception as e:
            return f"error reading the file {e}"
        
    else:
        return "Unsupported file format. Please upload a PDF or DOCX file."

# %%
model = joblib.load("model.pkl")
vect = joblib.load("vectorizer.pkl")
encoder = joblib.load("LEencoder.pkl")

# %%
def predict_roles(model, vectorizer, encoder, text, top_n=5):
    vec = vectorizer.transform([text])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    top_indices = probs.argsort()[-top_n:][::-1]
    top_classes =classes[top_indices]
    top_probs = probs[top_indices]

    decoded_roles = {
        encoder.inverse_transform([int(role)])[0]: prob
        for role, prob in zip(top_classes, top_probs)
    }
    return decoded_roles

# %%
with open("skills.json")as f:
    job_skills = json.load(f)

# %%
st.title("Resume Job Role Predictor") # app title
uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"]) # file uploader widget

if uploaded_file is not None:
    text= extract_file(uploaded_file)
    if 'error' not in text.lower() and 'unsupported' not in text.lower():
        st.subheader('Extracted Resume Text')
        st.write(text[:500] + '...' if len(text) >500 else text)
        processed = preprocess(text)
        prediction = predict_roles(model,vect,encoder,processed)
        st.subheade('Top Predicted Roles')
        for role,prob in prediction.items():
            st.write(f'**{role}**:{prob:.2f}')

            predictions = predict_roles(model, vect, encoder, processed)

            st.subheader("Top Predicted Roles")
            for role, prob in predictions.items():
                st.write(f"**{role}** :{prob:.2f}")

            best_role = max(predictions, key=predictions.get)
            if best_role in job_skills:
                st.subheader(f"Required skills for {best_role}")
                st.write(", ".join(job_skills[best_role]))

    else:
        st.error(text)


