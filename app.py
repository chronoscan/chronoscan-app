import streamlit as st
from PIL import Image
import json
from ultralytics import YOLO
import datetime

# Charger le modèle YOLOv8 pré-entraîné
model = YOLO('yolov8n.pt')

# Charger la base de données des périodes d'objets
with open("object_dates.json", "r") as f:
    object_periods = json.load(f)

st.title("📸 ChronoScan – Détection d'anachronismes")
st.write("Téléverse une image, choisis une époque, et je te dirai si des objets sont hors du temps.")

# Sélection de l'époque cible
target_year = st.number_input("Année de référence", min_value=1000, max_value=2100, value=1920)

# Téléversement de l'image
uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image téléversée", use_column_width=True)

    # Exécution de la détection d'objets
    st.write("🔍 Analyse en cours...")
    results = model.predict(image)

    # Résultats
    detected_labels = results[0].names
    detected_objects = results[0].boxes.cls.tolist()
    all_detected = [detected_labels[int(idx)] for idx in detected_objects]

    st.subheader("Objets détectés :")
    st.write(", ".join(set(all_detected)))

    st.subheader("Analyse des anachronismes :")
    for obj in set(all_detected):
        if obj in object_periods:
            start, end = object_periods[obj]
            if not (start <= target_year <= end):
                st.error(f"❌ Anachronisme : **{obj}** n'existait pas en {target_year} (existe entre {start}–{end})")
            else:
                st.success(f"✅ {obj} est cohérent avec l'époque.")
        else:
            st.warning(f"⚠️ Objet détecté **{obj}** : période inconnue dans la base.")
