from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS
import google.generativeai as genai
import re

app = Flask(__name__)
CORS(app)

# 🔹 Google Gemini yapılandırması
GOOGLE_API_KEY = "AIzaSyA3wqV06gjwGPrdRxIIaYlLQICExCr2Yoo"
genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel("gemini-2.0-flash")

# 🔹 Model yükleniyor
model = joblib.load('D:/bitirme/models/svc_model3.pkl')

# 🔹 Açıklama verisi
ds = pd.read_csv('D:/bitirme/symptom_Description.csv')
ds.index = ds['Disease']
ds = ds.drop('Disease', axis=1)

# 🔹 Önlem verisi
pr = pd.read_csv('D:/bitirme/symptom_precaution.csv').fillna("")
pr['precautions'] = pr[["Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4"]].apply(lambda row: ', '.join(filter(None, row)), axis=1)
pr = pr.drop([f"Precaution_{i}" for i in range(1, 5)], axis=1)
pr = pr.drop_duplicates()
pr.index = pr['Disease']
pr = pr.drop('Disease', axis=1)

# 🔹 Ana tahmin fonksiyonu
def predict_top5_diseases(user_symptoms):
    df1 = pd.read_csv("D:/bitirme/Symptom-severity.csv")
    symptom_weights = df1.set_index("Symptom")["weight"].to_dict()
    columns = [f"Symptom_{i}" for i in range(1, 18)]

    input_vector = [0] * len(columns)
    for i, sym in enumerate(user_symptoms[:17]):
        input_vector[i] = symptom_weights.get(sym, 0)

    input_vector = np.array(input_vector).reshape(1, -1)
    probs = model.predict_proba(input_vector)[0]
    top5_indices = np.argsort(probs)[-5:][::-1]
    possible_diseases = model.classes_[top5_indices]
    probabilities = probs[top5_indices]

    ds_dict = ds["Description"].to_dict()
    pr_dict = pr["precautions"].to_dict()

    results = []
    for disease, probability in zip(possible_diseases, probabilities):
        results.append({
            "disease": disease,
            "probability": round(probability * 100, 2),
            "description": ds_dict.get(disease, "No description available"),
            "precautions": pr_dict.get(disease, "No precautions available")
        })
    return results

# 🔹 Semptom çıkarımı
def extract_symptoms_from_text(text):
    prompt = f"""
Aşağıda bir kullanıcının sağlık şikayeti yer alıyor. Lütfen sadece bu 132 semptomdan hangilerini içerdiğini Python list formatında döndür.

Semptomlar: abdominal_pain, abnormal_menstruation, acidity, acute_liver_failure, altered_sensorium, anxiety, back_pain, belly_pain, blackheads, bladder_discomfort, blister, blood_in_sputum, bloody_stool, blurred_and_distorted_vision, breathlessness, brittle_nails, bruising, burning_micturition, chest_pain, chills, cold_hands_and_feets, coma, congestion, constipation, continuous_feel_of_urine, continuous_sneezing, cough, cramps, dark_urine, dehydration, depression, diarrhoea, dischromic_patches, distention_of_abdomen, dizziness, drying_and_tingling_lips, enlarged_thyroid, excessive_hunger, extra_marital_contacts, family_history, fast_heart_rate, fatigue, fluid_overload, foul_smell_ofurine, headache, high_fever, hip_joint_pain, history_of_alcohol_consumption, increased_appetite, indigestion, inflammatory_nails, internal_itching, irregular_sugar_level, irritability, irritation_in_anus, itching, joint_pain, knee_pain, lack_of_concentration, lethargy, loss_of_appetite, loss_of_balance, loss_of_smell, malaise, mild_fever, mood_swings, movement_stiffness, mucoid_sputum, muscle_pain, muscle_wasting, muscle_weakness, nausea, neck_pain, nodal_skin_eruptions, obesity, pain_behind_the_eyes, pain_during_bowel_movements, pain_in_anal_region, painful_walking, palpitations, passage_of_gases, patches_in_throat, phlegm, polyuria, prognosis, prominent_veins_on_calf, puffy_face_and_eyes, pus_filled_pimples, receiving_blood_transfusion, receiving_unsterile_injections, red_sore_around_nose, red_spots_over_body, redness_of_eyes, restlessness, runny_nose, rusty_sputum, scurring, shivering, silver_like_dusting, sinus_pressure, skin_peeling, skin_rash, slurred_speech, small_dents_in_nails, spinning_movements, spotting_urination, stiff_neck, stomach_bleeding, stomach_pain, sunken_eyes, sweating, swelled_lymph_nodes, swelling_joints, swelling_of_stomach, swollen_blood_vessels, swollen_extremeties, swollen_legs, throat_irritation, toxic_look_(typhos), ulcers_on_tongue, unsteadiness, visual_disturbances, vomiting, watering_from_eyes, weakness_in_limbs, weakness_of_one_body_side, weight_gain, weight_loss, yellow_crust_ooze, yellow_urine, yellowing_of_eyes, yellowish_skin

Cümle: {text}
Cevap (sadece Python listesi):
"""
    try:
        response = llm_model.generate_content(prompt)
        result_text = response.text.strip()
        match = re.search(r"\[.*\]", result_text)
        if match:
            return eval(match.group())
    except Exception as e:
        print(f"Extraction error: {e}")
    return []

# 🔹 Medikal içerik kontrolü
def is_medical_question(text):
    prompt = f"""
Aşağıdaki cümle bir tıbbi semptom, hastalık, tedavi veya sağlıkla ilgili mi? Sadece 'evet' ya da 'hayır' olarak cevap ver.

Cümle: "{text}"
Cevap:"""
    try:
        response = llm_model.generate_content(prompt)
        answer = response.text.lower().strip()
        return "evet" in answer
    except Exception as e:
        print(f"Medical check error: {e}")
    return False

# 🔹 Konuşma durumu
dialogue_state = {"messages": [], "symptoms": []}

# 🔹 Chatbot endpoint
@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Mesaj boş olamaz."}), 400

    if not is_medical_question(message):
        reply = "Bu asistan yalnızca tıbbi konulara yardımcı olabilir. Lütfen sağlıkla ilgili bir şey sorun."
        dialogue_state["messages"].append({"user": message, "bot": reply})
        return jsonify({"message": reply})

    new_symptoms = extract_symptoms_from_text(message)

    if new_symptoms:
        dialogue_state["symptoms"] = list(set(dialogue_state["symptoms"] + new_symptoms))
        prediction = predict_top5_diseases(dialogue_state["symptoms"])
        dialogue_state["last_disease"] = prediction[0]['disease']  # 🔹 Güncelle
        reply = f"Semptomlarınıza göre en olası hastalık: {prediction[0]['disease']}\n\nAçıklama: {prediction[0]['description']}\nÖnlemler: {prediction[0]['precautions']}"
    else:
        # 🔹 Eğer önce hastalık tahmini yapılmışsa, onun üzerinden konuşmaya devam et
        if dialogue_state["last_disease"]:
            prompt = f"""Kullanıcının daha önce semptomlarına göre tahmin edilen hastalık: {dialogue_state['last_disease']}
Şimdi bu hastalık hakkında soru soruyor: "{message}"
Lütfen bu hastalık bağlamında cevap ver."""
            response = llm_model.generate_content(prompt)
            reply = response.text.strip()
        else:
            # Henüz hastalık tahmini yapılmamışsa, normal yanıt ver
            response = llm_model.generate_content(message)
            reply = response.text.strip()

    dialogue_state["messages"].append({"user": message, "bot": reply})
    return jsonify({"message": reply})


# 🔹 Manuel tahmin endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", [])

    if isinstance(symptoms, str):
        symptoms = [s.strip() for s in symptoms.split(",")]

    if not symptoms or not isinstance(symptoms, list):
        return jsonify({"error": "Semptom listesi geçerli değil."}), 400

    try:
        result = predict_top5_diseases(symptoms)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
