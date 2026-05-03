import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from transformers import pipeline, CLIPProcessor, CLIPModel


st.set_page_config(
    page_title="Havalimanı Deneyimi Karar Destek Demo",
    layout="wide"
)

st.title("Havalimanı Yolcu Deneyimi Karar Destek Demo")

sayfa = st.sidebar.radio(
    "Ekran Seçiniz",
    ["1. Veri Seti Analizi", "2. Manuel Yorum ve Görsel Analizi"]
)


@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device


def load_image_from_github(url):
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content)).convert("RGB")
    except:
        return None
    return None


def get_sentiment(text):
    model = load_sentiment_model()
    result = model(str(text)[:512])[0]
    return result["score"] if result["label"] == "POSITIVE" else -result["score"]


def classify_image(image):
    model, processor, device = load_clip_model()

    labels = {
        "security_area": "a photo of airport security screening area or passport control",
        "waiting_area": "a photo of airport waiting area with seats and passengers",
        "boarding_gate": "a photo of an airport boarding gate",
        "baggage_claim": "a photo of airport baggage claim area",
        "food_retail_area": "a photo of airport food court restaurant cafe or retail shop",
        "restroom": "a photo of airport restroom or toilet facilities",
        "terminal_general": "a photo of airport terminal interior",
        "unclear": "an unclear or irrelevant airport photo"
    }

    label_names = list(labels.keys())
    prompts = list(labels.values())

    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

    best = probs.argmax()
    return label_names[best], float(probs[best])


def run_bertopic(docs, min_topic_size=2):
    stopwords = [
        "the", "and", "to", "of", "in", "is", "it", "for", "was", "are", "you",
        "this", "that", "with", "as", "on", "at", "be", "have", "has", "had",
        "we", "they", "he", "she", "my", "our", "your", "their",
        "a", "an", "do", "does", "did", "done",
        "there", "here", "where", "when", "which", "who", "what",
        "if", "even", "but", "all", "very", "still", "been",
        "not", "no", "don", "dont", "can", "could", "would", "should",
        "airport", "istanbul", "iga", "ist", "flight", "flights",
        "turkish", "verified", "unverified", "trip", "review"
    ]

    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        ngram_range=(1, 2),
        min_df=1
    )

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    topic_model = BERTopic(
        embedding_model=embedding_model,
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=False,
        verbose=False,
        min_topic_size=min_topic_size
    )

    topics, _ = topic_model.fit_transform(docs)
    return topic_model, topics


def auto_label_topic(words):
    joined = " ".join(words)

    if any(k in joined for k in ["security", "passport", "check", "control", "departure"]):
        return "Güvenlik ve check-in süreçleri"
    elif any(k in joined for k in ["food", "internet", "toilet", "wifi", "expensive", "free"]):
        return "Yiyecek-içecek, internet ve tuvalet hizmetleri"
    elif any(k in joined for k in ["staff", "passenger", "service", "rude"]):
        return "Personel ve yolcu deneyimi"
    elif any(k in joined for k in ["seat", "waiting", "gate", "crowd"]):
        return "Bekleme alanı ve terminal konforu"
    elif any(k in joined for k in ["baggage", "luggage", "bag"]):
        return "Bagaj hizmetleri"
    else:
        return "Diğer"


def manual_criterion_mapping(text):
    text = text.lower()

    if any(k in text for k in ["security", "passport", "check-in", "check in", "control"]):
        return "Güvenlik ve check-in süreçleri"
    elif any(k in text for k in ["food", "internet", "wifi", "toilet", "restroom", "expensive"]):
        return "Yiyecek-içecek, internet ve tuvalet hizmetleri"
    elif any(k in text for k in ["staff", "rude", "employee", "personnel", "service"]):
        return "Personel ve yolcu deneyimi"
    elif any(k in text for k in ["seat", "waiting", "queue", "gate", "crowded"]):
        return "Bekleme alanı ve terminal konforu"
    elif any(k in text for k in ["baggage", "luggage", "bag"]):
        return "Bagaj hizmetleri"
    else:
        return "Diğer"


def get_image_files(text):
    if pd.isna(text):
        return []
    return [x.strip() for x in str(text).split("|") if x.strip()]


def build_image_dict_from_github(df, image_base_url):
    image_dict = {}

    if "image_files" not in df.columns or not image_base_url:
        return image_dict

    for files in df["image_files"].dropna():
        for file in str(files).split("|"):
            file = file.strip()

            if file and file not in image_dict:
                image_url = f"{image_base_url}/{file}"
                image = load_image_from_github(image_url)

                if image is not None:
                    image_dict[file] = image

    return image_dict


def analyse_images_for_review(image_files, image_dict):
    labels = []
    confidences = []

    for img_file in image_files:
        if img_file not in image_dict:
            continue

        try:
            image = image_dict[img_file]
            label, conf = classify_image(image)
            labels.append(label)
            confidences.append(conf)
        except:
            continue

    return labels, confidences


def criterion_visual_match(criterion, image_labels, image_confidences):
    criterion_image_map = {
        "Güvenlik ve check-in süreçleri": ["security_area", "boarding_gate"],
        "Yiyecek-içecek, internet ve tuvalet hizmetleri": ["food_retail_area", "restroom"],
        "Personel ve yolcu deneyimi": ["waiting_area", "terminal_general"],
        "Bekleme alanı ve terminal konforu": ["waiting_area", "terminal_general", "boarding_gate"],
        "Bagaj hizmetleri": ["baggage_claim"]
    }

    allowed = criterion_image_map.get(criterion, [])

    matched_conf = [
        conf for label, conf in zip(image_labels, image_confidences)
        if label in allowed
    ]

    if len(matched_conf) == 0:
        return 0.0

    return sum(matched_conf) / len(matched_conf)


def build_mcdm(df):
    summary = df.groupby("criterion").agg(
        frequency=("content", "count"),
        sentiment_mean=("sentiment", "mean"),
        visual_match_mean=("visual_match", "mean")
    ).reset_index()

    summary["sentiment_intensity"] = summary["sentiment_mean"].abs()
    summary["v_j"] = 1 + summary["visual_match_mean"].fillna(0)

    summary["raw_score"] = (
        summary["frequency"] *
        summary["sentiment_intensity"] *
        summary["v_j"]
    )

    if summary["raw_score"].sum() > 0:
        summary["weight"] = summary["raw_score"] / summary["raw_score"].sum()
    else:
        summary["weight"] = 0

    for col in ["frequency", "sentiment_intensity", "v_j"]:
        min_val = summary[col].min()
        max_val = summary[col].max()

        if max_val == min_val:
            summary[col + "_norm"] = 1
        else:
            summary[col + "_norm"] = (summary[col] - min_val) / (max_val - min_val)

    summary["mcdm_score"] = (
        summary["frequency_norm"] * 0.40 +
        summary["sentiment_intensity_norm"] * 0.40 +
        summary["v_j_norm"] * 0.20
    )

    summary = summary.sort_values("mcdm_score", ascending=False).reset_index(drop=True)

    summary["rank"] = summary["mcdm_score"].rank(
        ascending=False,
        method="dense"
    ).astype(int)

    return summary


# ==================================================
# EKRAN 1
# ==================================================

if sayfa == "1. Veri Seti Analizi":

    st.header("1. Veri Seti Analizi")

    st.info("""
    Bu ekranda GitHub'daki Excel veri seti ve görseller kullanılarak konu modelleme,
    duygu analizi, görsel sınıflandırma ve MCDM sonuçları üretilir.
    """)

    excel_url = st.text_input(
        "GitHub Excel Raw URL",
        value="https://raw.githubusercontent.com/bahribarankocak/airport-dashboard/main/reviews.xlsx"
    )

    image_base_url = st.text_input(
        "GitHub Görsel Klasörü Base URL",
        value="https://raw.githubusercontent.com/bahribarankocak/airport-dashboard/main/images"
    )

    min_topic_size = st.slider(
        "Minimum konu büyüklüğü",
        min_value=2,
        max_value=10,
        value=2
    )

    if st.button("Veriyi GitHub'dan Yükle ve Analizi Başlat"):

        try:
            df = pd.read_excel(excel_url)
            st.success("Excel GitHub'dan başarıyla yüklendi.")
        except Exception as e:
            st.error(f"Excel yüklenemedi. URL'yi kontrol edin. Hata: {e}")
            st.stop()

        if "content" not in df.columns:
            st.error("Excel dosyasında 'content' kolonu olmalıdır.")
            st.stop()

        df = df[df["content"].notna()].copy()
        df["content"] = df["content"].astype(str).str.strip()
        df = df[df["content"].str.len() > 20].reset_index(drop=True)

        if len(df) < 3:
            st.error("Analiz için yeterli yorum yok.")
            st.stop()

        with st.spinner("GitHub görselleri kontrol ediliyor..."):
            image_dict = build_image_dict_from_github(df, image_base_url)

        use_images = False

        if "image_files" in df.columns and len(image_dict) > 0:
            use_images = True
            st.success(f"Görsel analiz aktif. Yüklenen görsel sayısı: {len(image_dict)}")
        else:
            st.warning("Görsel veri bulunamadı. Sistem sadece metin analizi modunda çalışacak.")
            df["image_files"] = ""
            df["image_labels"] = ""
            df["image_confidence_avg"] = 0
            df["visual_match"] = 0

        st.subheader("Yüklenen Veri")
        st.dataframe(df.head())

        docs = df["content"].tolist()

        with st.spinner("Konu modelleme yapılıyor..."):
            topic_model, topics = run_bertopic(docs, min_topic_size)
            df["topic"] = topics

        topic_info = topic_model.get_topic_info()

        topic_label_map = {}

        for topic_id in topic_info["Topic"].tolist():
            if topic_id == -1:
                continue

            words = topic_model.get_topic(topic_id)
            topic_words = [w[0] for w in words[:8]]
            topic_label_map[topic_id] = auto_label_topic(topic_words)

        df["criterion"] = df["topic"].map(topic_label_map)
        df = df[df["criterion"].notna()].reset_index(drop=True)

        with st.spinner("Duygu analizi yapılıyor..."):
            df["sentiment"] = df["content"].apply(get_sentiment)

        if use_images:
            with st.spinner("Görseller analiz ediliyor..."):
                all_labels = []
                all_confidences = []
                visual_matches = []

                for _, row in df.iterrows():
                    files = get_image_files(row["image_files"])
                    labels, confidences = analyse_images_for_review(files, image_dict)

                    all_labels.append(" | ".join(labels))

                    all_confidences.append(
                        sum(confidences) / len(confidences)
                        if len(confidences) > 0
                        else 0
                    )

                    visual_matches.append(
                        criterion_visual_match(
                            row["criterion"],
                            labels,
                            confidences
                        )
                    )

                df["image_labels"] = all_labels
                df["image_confidence_avg"] = all_confidences
                df["visual_match"] = visual_matches
        else:
            df["image_labels"] = ""
            df["image_confidence_avg"] = 0
            df["visual_match"] = 0

        mcdm_df = build_mcdm(df)

        st.subheader("Konu Özeti")
        st.dataframe(topic_info)

        st.subheader("Konu → Kriter Eşleştirme")
        st.write(topic_label_map)

        st.subheader("Analiz Edilmiş Yorumlar")
        display_cols = [
            "content", "topic", "criterion", "sentiment",
            "image_files", "image_labels", "visual_match"
        ]
        existing_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[existing_cols])

        st.subheader("MCDM Sonuçları")
        st.dataframe(mcdm_df)

        if len(mcdm_df) > 0:
            top_row = mcdm_df.sort_values("rank").iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Kriter Sayısı", len(mcdm_df))
            c2.metric("En Öncelikli Alan", top_row["criterion"])
            c3.metric("Öncelik Skoru", round(top_row["mcdm_score"], 3))

            st.subheader("Kriter Ağırlıkları")

            fig, ax = plt.subplots()
            chart_df = mcdm_df.sort_values("weight", ascending=True)
            ax.barh(chart_df["criterion"], chart_df["weight"])
            ax.set_xlabel("Ağırlık")
            ax.set_title("Veri Tabanlı Kriter Ağırlıkları")
            st.pyplot(fig)

            st.subheader("İyileştirme Öncelik Skorları")

            fig2, ax2 = plt.subplots()
            score_df = mcdm_df.sort_values("mcdm_score", ascending=True)
            ax2.barh(score_df["criterion"], score_df["mcdm_score"])
            ax2.set_xlabel("MCDM Skoru")
            ax2.set_title("Hizmet İyileştirme Öncelikleri")
            st.pyplot(fig2)

            st.subheader("Yönetimsel Yorum")

            st.write(
                f"Analiz sonuçlarına göre en öncelikli hizmet alanı "
                f"**{top_row['criterion']}** olarak belirlenmiştir. "
                f"Bu alan, havalimanı yöneticileri açısından ilk iyileştirme "
                f"odaklarından biri olarak değerlendirilebilir."
            )

        st.download_button(
            "Analiz Edilmiş Veriyi İndir",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="analiz_edilmis_yorumlar.csv",
            mime="text/csv"
        )

        st.download_button(
            "MCDM Sonuçlarını İndir",
            data=mcdm_df.to_csv(index=False).encode("utf-8"),
            file_name="mcdm_sonuclari.csv",
            mime="text/csv"
        )


# ==================================================
# EKRAN 2
# ==================================================

elif sayfa == "2. Manuel Yorum ve Görsel Analizi":

    st.header("2. Manuel Yorum ve Görsel Analizi")

    manual_text = st.text_area(
        "Yolcu yorumunu giriniz",
        height=150,
        placeholder="Örnek: Security queue was very long but the staff were helpful."
    )

    uploaded_image = st.file_uploader(
        "Bir görsel yükleyiniz (opsiyonel)",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if st.button("Manuel Analizi Başlat"):

        if not manual_text.strip():
            st.error("Lütfen bir yorum giriniz.")
        else:
            sentiment = get_sentiment(manual_text)
            criterion = manual_criterion_mapping(manual_text)

            image_label = "Görsel yok"
            image_confidence = 0.0

            if uploaded_image is not None:
                image = Image.open(uploaded_image).convert("RGB")
                st.image(image, caption="Yüklenen Görsel", use_container_width=True)

                with st.spinner("Görsel sınıflandırılıyor..."):
                    image_label, image_confidence = classify_image(image)

            st.subheader("Manuel Analiz Sonucu")

            c1, c2, c3 = st.columns(3)
            c1.metric("Tespit Edilen Kriter", criterion)
            c2.metric("Duygu Skoru", round(sentiment, 3))
            c3.metric("Görsel Etiketi", image_label)

            st.json({
                "yorum": manual_text,
                "kriter": criterion,
                "duygu_skoru": sentiment,
                "gorsel_etiketi": image_label,
                "gorsel_guven_skoru": image_confidence
            })
