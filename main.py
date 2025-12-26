import streamlit as st
import re
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('stopwords')
from nltk.corpus import stopwords

# ===============================
# Cargar modelo BETO
# ===============================
MODEL_NAME = "finiteautomata/beto-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = ["negativo", "neutral", "positivo"]
stop_words = set(stopwords.words("spanish"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-záéíóúñ\s]", "", text)
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text.strip()

def analyze_sentiment(text):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)
    return labels[prediction.item()], float(confidence.item())

# ===============================
# Interfaz Streamlit
# ===============================
st.title("SentiRest - Análisis de Reseñas")
st.write("Simulación de reseñas y análisis de sentimientos usando BETO")

# Simulación de reseñas
simulated_reviews = [
    "La comida estuvo deliciosa y el servicio excelente.",
    "El lugar es bonito, pero el tiempo de espera fue muy largo.",
    "No me gustó la atención, muy descorteses.",
    "Excelente experiencia, volvería sin dudarlo.",
    "Comida regular, nada fuera de lo común.",
    "El ambiente es muy acogedor y agradable.",
    "La pizza estaba fría y poco sabrosa.",
    "El personal fue muy atento y amable.",
    "La relación calidad-precio es buena.",
    "No recomiendo este lugar, mala experiencia.",
    "Me encantó la decoración y la música del local.",
    "Los postres son increíbles, muy recomendados.",
    "Demoraron mucho en traer los platos.",
    "Todo estaba perfecto, desde la entrada hasta el postre.",
    "No volvería, la comida estaba salada y sin sabor.",
    "Servicio rápido y eficiente, muy buena atención.",
    "La terraza es perfecta para reuniones familiares.",
    "Precios elevados para la calidad que ofrecen.",
    "Excelente menú vegetariano y opciones saludables.",
    "Muy buena ubicación, fácil de encontrar y estacionar."
]

if st.button("Analizar Reseñas"):
    results = []
    for review in simulated_reviews:
        sentiment, confidence = analyze_sentiment(review)
        results.append({"text": review, "sentiment": sentiment, "confidence": confidence})
    
    # Mostrar resultados
    st.subheader("Reseñas Analizadas")
    for r in results:
        st.write(f"{r['text']} → {r['sentiment']} ({r['confidence']*100:.1f}%)")
    
    # Conteo por sentimiento
    counts = {"positivo": 0, "neutral": 0, "negativo": 0}
    for r in results:
        counts[r["sentiment"]] += 1
    st.subheader("Cantidad por Sentimiento")
    st.write(counts)
    
    # Gráfica circular
    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=counts.keys(), autopct='%1.1f%%', colors=["#4CAF50", "#FFC107", "#F44336"])
    st.pyplot(fig)
    
    # Nube de palabras
    all_text = " ".join([r["text"] for r in results])
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
    st.subheader("Nube de palabras")
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)
