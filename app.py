import re
import numpy as np
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


# ==================================================
# SAYFA AYARLARI
# ==================================================

st.set_page_config(
    page_title="Havalimanı Yolcu Deneyimi Karar Destek Prototipi",
    layout="wide"
)

st.title("Havalimanı Yolcu Deneyimi Karar Destek Prototipi")
st.caption(
    "UGC → Önişleme → Konu Modelleme → Duygu Analizi → Görüntü İşleme → "
    "Karar Matrisi / CRITIC → TOPSIS"
)

sayfa = st.sidebar.radio(
    "Ekran Seçiniz",
    ["1. Veri Seti Analizi", "2. Manuel Yorum ve Görsel Analizi"]
)

with st.sidebar.expander("Proje metodolojik akışı", expanded=True):
    st.markdown(
        """
        1. **Veri**  
        2. **Önişleme**  
        3. **Konu modelleme**  
        4. **Duygu analizi**  
        5. **Görüntü işleme**  
        6. **Karar matrisi + CRITIC**  
        7. **TOPSIS karar modelleme**
        """
    )


# ==================================================
# MODEL YÜKLEME
# ==================================================

@st.cache_resource
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


@st.cache_resource
def load_clip_model():
    """Ön prototip için temas noktası sınıflandırması.

    Nihai projede CNN/ConvMixer/MLP-Mixer/ViT/Swin Transformer karşılaştırması
    ve özgün model geliştirme aşaması bunun yerini alacaktır.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device


# ==================================================
# ADIM 1-2: VERİ VE ÖNİŞLEME
# ==================================================

def load_image_from_github(url):
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return Image.open(BytesIO(r.content)).convert("RGB")
    except Exception:
        return None
    return None


def preprocess_topic_text(text):
    """BERTopic için temel metin temizliği.

    Ham metin ayrıca korunur. Stop-word filtreleme CountVectorizer aşamasında
    uygulanır. Nihai araştırma uygulamasında lemmatizasyon ayrıca eklenebilir.
    """
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==================================================
# ADIM 3: KONU MODELLEME
# ==================================================

def run_bertopic(docs, min_topic_size=2):
    # Olumsuzluk belirteçlerini (not/no vb.) stop-word listesine koymuyoruz.
    stopwords = [
        "the", "and", "to", "of", "in", "is", "it", "for", "was", "are", "you",
        "this", "that", "with", "as", "on", "at", "be", "have", "has", "had",
        "we", "they", "he", "she", "my", "our", "your", "their",
        "a", "an", "do", "does", "did", "done",
        "there", "here", "where", "when", "which", "who", "what",
        "if", "even", "but", "all", "very", "still", "been",
        "can", "could", "would", "should",
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
    """Demo amaçlı konu → hizmet alanı etiketleme.

    Konular BERTopic ile veri tabanlı keşfedilir; bu fonksiyon yalnızca bulunan
    konulara okunabilir hizmet alanı adı vermek için kullanılır.
    """
    joined = " ".join(words).lower()

    if any(k in joined for k in ["security", "passport", "screening", "control"]):
        return "Güvenlik ve pasaport kontrolü"
    if any(k in joined for k in ["baggage", "luggage", "bag", "carousel"]):
        return "Bagaj hizmetleri"
    if any(k in joined for k in ["seat", "waiting", "gate", "crowd", "boarding"]):
        return "Bekleme alanı ve biniş kapısı"
    if any(k in joined for k in ["food", "restaurant", "cafe", "shop", "retail", "price", "expensive"]):
        return "Yiyecek-içecek ve perakende"
    if any(k in joined for k in ["toilet", "restroom", "clean", "dirty", "hygiene"]):
        return "Tuvalet ve temizlik"
    if any(k in joined for k in ["staff", "employee", "personnel", "rude", "helpful", "service"]):
        return "Personel hizmetleri"
    if any(k in joined for k in ["wifi", "internet"]):
        return "Dijital hizmetler"
    if any(k in joined for k in ["check-in", "checkin", "check"]):
        return "Check-in süreçleri"

    return "Diğer"


def manual_service_area_mapping(text):
    text = str(text).lower()

    if any(k in text for k in ["security", "passport", "screening", "control"]):
        return "Güvenlik ve pasaport kontrolü"
    if any(k in text for k in ["baggage", "luggage", "bag", "carousel"]):
        return "Bagaj hizmetleri"
    if any(k in text for k in ["seat", "waiting", "queue", "gate", "crowded", "boarding"]):
        return "Bekleme alanı ve biniş kapısı"
    if any(k in text for k in ["food", "restaurant", "cafe", "shop", "retail", "expensive", "price"]):
        return "Yiyecek-içecek ve perakende"
    if any(k in text for k in ["toilet", "restroom", "clean", "dirty", "hygiene"]):
        return "Tuvalet ve temizlik"
    if any(k in text for k in ["staff", "rude", "employee", "personnel", "helpful", "service"]):
        return "Personel hizmetleri"
    if any(k in text for k in ["wifi", "internet"]):
        return "Dijital hizmetler"
    if any(k in text for k in ["check-in", "check in", "checkin"]):
        return "Check-in süreçleri"

    return "Diğer"


# ==================================================
# ADIM 4: DUYGU ANALİZİ
# ==================================================

def get_sentiment_scores(text):
    """İmzalı duygu ve iyileştirme modeli için olumsuzluk skoru döndürür.

    signed_sentiment: pozitif için +p, negatif için -p
    negative_score: negatif yorumda p, pozitif yorumda 0
    """
    model = load_sentiment_model()
    result = model(str(text)[:512])[0]

    label = str(result["label"]).upper()
    score = float(result["score"])

    signed_sentiment = score if label == "POSITIVE" else -score
    negative_score = score if label == "NEGATIVE" else 0.0

    return signed_sentiment, negative_score


# ==================================================
# ADIM 5: GÖRÜNTÜ İŞLEME / ÇOK MODLU DESTEK
# ==================================================

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

    best = int(probs.argmax())
    return label_names[best], float(probs[best])


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
                image_url = f"{image_base_url.rstrip('/')}/{file}"
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
        except Exception:
            continue

    return labels, confidences


def service_area_multimodal_support(service_area, image_labels, image_confidences):
    """Metindeki hizmet alanının görsel temas noktasıyla bağlamsal desteği.

    Bu skor görseldeki sorunun şiddetini ölçmez.
    """
    service_area_image_map = {
        "Güvenlik ve pasaport kontrolü": ["security_area"],
        "Check-in süreçleri": ["terminal_general"],
        "Bekleme alanı ve biniş kapısı": ["waiting_area", "boarding_gate", "terminal_general"],
        "Bagaj hizmetleri": ["baggage_claim"],
        "Yiyecek-içecek ve perakende": ["food_retail_area"],
        "Tuvalet ve temizlik": ["restroom"],
        "Personel hizmetleri": ["terminal_general", "waiting_area", "boarding_gate", "security_area"],
        "Dijital hizmetler": ["terminal_general", "waiting_area"]
    }

    allowed = service_area_image_map.get(service_area, [])

    matched_conf = [
        conf for label, conf in zip(image_labels, image_confidences)
        if label in allowed
    ]

    if len(matched_conf) == 0:
        return 0.0

    return float(np.mean(matched_conf))


# ==================================================
# ADIM 6: KARAR MATRİSİ, NORMALİZASYON, CRITIC
# ==================================================

def minmax_benefit_normalize(series):
    series = pd.to_numeric(series, errors="coerce").fillna(0).astype(float)
    min_val = series.min()
    max_val = series.max()
    if np.isclose(max_val, min_val):
        # Sabit kriter bilgi taşımadığı için CRITIC'te 0'a çekilir.
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - min_val) / (max_val - min_val)


def build_decision_matrix(df):
    """Hizmet alanları = alternatifler; C1-C3 = karar kriterleri."""
    work = df[df["service_area"].notna()].copy()
    work = work[work["service_area"] != "Diğer"].copy()

    if work.empty:
        return pd.DataFrame()

    total_reviews = len(work)

    summary = work.groupby("service_area").agg(
        mention_count=("content", "count"),
        negative_sentiment=("negative_score", "mean"),
        multimodal_support=("multimodal_support", "mean")
    ).reset_index()

    summary["prevalence"] = summary["mention_count"] / total_reviews

    return summary[
        ["service_area", "mention_count", "prevalence", "negative_sentiment", "multimodal_support"]
    ]


def normalize_decision_matrix(decision_df):
    normalized = decision_df[["service_area"]].copy()
    normalized["C1_prevalence"] = minmax_benefit_normalize(decision_df["prevalence"])
    normalized["C2_negative_sentiment"] = minmax_benefit_normalize(decision_df["negative_sentiment"])
    normalized["C3_multimodal_support"] = minmax_benefit_normalize(decision_df["multimodal_support"])
    return normalized


def calculate_critic_weights(normalized_df):
    criteria = [
        "C1_prevalence",
        "C2_negative_sentiment",
        "C3_multimodal_support"
    ]

    X = normalized_df[criteria].astype(float)
    std = X.std(ddof=0)

    active = std[std > 1e-12].index.tolist()
    weights = pd.Series(0.0, index=criteria)
    info = pd.Series(0.0, index=criteria)

    if len(active) == 0:
        weights[:] = 1.0 / len(criteria)
    elif len(active) == 1:
        weights[active[0]] = 1.0
        info[active[0]] = std[active[0]]
    else:
        corr = X[active].corr().fillna(0.0)
        conflict = pd.Series(index=active, dtype=float)

        for col in active:
            # Diagonal terim 1-1=0 olduğundan toplamı değiştirmez.
            conflict[col] = (1.0 - corr.loc[col, active]).sum()

        critic_information = std[active] * conflict

        if critic_information.sum() <= 1e-12:
            weights[active] = 1.0 / len(active)
            info[active] = std[active]
        else:
            weights[active] = critic_information / critic_information.sum()
            info[active] = critic_information

    label_map = {
        "C1_prevalence": "C1 - Yaygınlık",
        "C2_negative_sentiment": "C2 - Olumsuz duygu",
        "C3_multimodal_support": "C3 - Çok modlu destek"
    }

    return pd.DataFrame({
        "criterion": criteria,
        "criterion_name": [label_map[c] for c in criteria],
        "critic_information": [float(info[c]) for c in criteria],
        "weight": [float(weights[c]) for c in criteria]
    })


# ==================================================
# ADIM 7: TOPSIS KARAR MODELLEME
# ==================================================

def calculate_topsis(decision_df, weights_df):
    """Tüm kriterler iyileştirme önceliği açısından fayda yönlü ele alınır."""
    if decision_df.empty:
        return pd.DataFrame()

    criteria_cols = ["prevalence", "negative_sentiment", "multimodal_support"]
    X = decision_df[criteria_cols].astype(float).to_numpy()

    # Vektör normalizasyonu (TOPSIS)
    denominators = np.sqrt((X ** 2).sum(axis=0))
    denominators = np.where(denominators == 0, 1.0, denominators)
    R = X / denominators

    weight_map = dict(zip(weights_df["criterion"], weights_df["weight"]))
    w = np.array([
        weight_map.get("C1_prevalence", 0.0),
        weight_map.get("C2_negative_sentiment", 0.0),
        weight_map.get("C3_multimodal_support", 0.0)
    ], dtype=float)

    if np.isclose(w.sum(), 0):
        w = np.array([1/3, 1/3, 1/3])
    else:
        w = w / w.sum()

    V = R * w

    # Üçü de yüksek olduğunda iyileştirme önceliği artıyor.
    ideal_best = V.max(axis=0)
    ideal_worst = V.min(axis=0)

    d_best = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    denom = d_best + d_worst
    score = np.divide(
        d_worst,
        denom,
        out=np.zeros_like(d_worst),
        where=denom != 0
    )

    result = decision_df.copy()
    result["topsis_score"] = score
    result = result.sort_values("topsis_score", ascending=False).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)

    return result


# ==================================================
# EKRAN 1: VERİ SETİ ANALİZİ
# ==================================================

if sayfa == "1. Veri Seti Analizi":

    st.header("1. Veri Seti Analizi")

    st.info(
        "Bu ekran proje metodolojisinin ön prototipidir. Görüntü sınıflandırmasında "
        "şimdilik CLIP kullanılmaktadır; proje kapsamında CNN tabanlı transfer öğrenimi, "
        "ConvMixer, MLP-Mixer, ViT ve Swin Transformer modelleri karşılaştırılarak "
        "özgün görüntü sınıflandırma modeli geliştirilecektir."
    )

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

    if st.button("Veriyi GitHub'dan Yükle ve Analizi Başlat", type="primary"):

        # ---------- ADIM 1: VERİ ----------
        st.subheader("Adım 1 — Veri")

        try:
            df = pd.read_excel(excel_url)
            st.success(f"Veri başarıyla yüklendi. Ham kayıt sayısı: {len(df)}")
        except Exception as e:
            st.error(f"Excel yüklenemedi. URL'yi kontrol edin. Hata: {e}")
            st.stop()

        if "content" not in df.columns:
            st.error("Excel dosyasında 'content' kolonu olmalıdır.")
            st.stop()

        # ---------- ADIM 2: ÖNİŞLEME ----------
        st.subheader("Adım 2 — Önişleme")

        df = df[df["content"].notna()].copy()
        df["content"] = df["content"].astype(str).str.strip()
        df = df.drop_duplicates(subset=["content"]).copy()
        df = df[df["content"].str.len() > 20].reset_index(drop=True)

        df["topic_text"] = df["content"].apply(preprocess_topic_text)
        df = df[df["topic_text"].str.len() > 10].reset_index(drop=True)

        if len(df) < 3:
            st.error("Analiz için yeterli yorum yok.")
            st.stop()

        st.write(f"Önişleme sonrası analiz edilebilir yorum sayısı: **{len(df)}**")
        st.dataframe(df[["content", "topic_text"]].head(10), use_container_width=True)

        with st.spinner("GitHub görselleri kontrol ediliyor..."):
            image_dict = build_image_dict_from_github(df, image_base_url)

        use_images = "image_files" in df.columns and len(image_dict) > 0

        if use_images:
            st.success(f"Görsel veri bulundu. Yüklenen görsel sayısı: {len(image_dict)}")
        else:
            st.warning(
                "Görsel veri bulunamadı. C3 - Çok modlu destek sabit kalacağından "
                "CRITIC bu kriteri bilgi taşımayan kriter olarak değerlendirebilir."
            )
            if "image_files" not in df.columns:
                df["image_files"] = ""

        # ---------- ADIM 3: KONU MODELLEME ----------
        st.subheader("Adım 3 — Konu Modelleme")

        docs = df["topic_text"].tolist()

        with st.spinner("Sentence-Transformer + BERTopic çalıştırılıyor..."):
            topic_model, topics = run_bertopic(docs, min_topic_size)
            df["topic"] = topics

        topic_info = topic_model.get_topic_info()
        topic_label_map = {}

        for topic_id in topic_info["Topic"].tolist():
            if topic_id == -1:
                continue

            words = topic_model.get_topic(topic_id)
            topic_words = [w[0] for w in words[:8]] if words else []
            topic_label_map[topic_id] = auto_label_topic(topic_words)

        df["service_area"] = df["topic"].map(topic_label_map)
        df = df[df["service_area"].notna()].reset_index(drop=True)

        st.caption(
            "Not: BERTopic konuları veri tabanlı olarak keşfeder. Prototipte konu adları "
            "okunabilirlik için anahtar sözcük tabanlı otomatik olarak etiketlenmektedir."
        )
        st.dataframe(topic_info, use_container_width=True)
        st.write("**Konu → Hizmet alanı eşleştirmesi:**", topic_label_map)

        # ---------- ADIM 4: DUYGU ANALİZİ ----------
        st.subheader("Adım 4 — Duygu Analizi")

        with st.spinner("Duygu analizi yapılıyor..."):
            sentiment_results = df["content"].apply(get_sentiment_scores)
            df["sentiment"] = sentiment_results.apply(lambda x: x[0])
            df["negative_score"] = sentiment_results.apply(lambda x: x[1])

        st.caption(
            "Karar modelinde mutlak duygu şiddeti yerine yalnızca olumsuzluk skoru "
            "kullanılır; güçlü pozitif yorumlar iyileştirme önceliğini artırmaz."
        )

        # ---------- ADIM 5: GÖRÜNTÜ İŞLEME ----------
        st.subheader("Adım 5 — Görüntü İşleme ve Çok Modlu Destek")

        if use_images:
            with st.spinner("Görsellerde havalimanı temas noktaları sınıflandırılıyor..."):
                all_labels = []
                all_confidences = []
                multimodal_supports = []

                for _, row in df.iterrows():
                    files = get_image_files(row.get("image_files", ""))
                    labels, confidences = analyse_images_for_review(files, image_dict)

                    all_labels.append(" | ".join(labels))
                    all_confidences.append(
                        float(np.mean(confidences)) if confidences else 0.0
                    )
                    multimodal_supports.append(
                        service_area_multimodal_support(
                            row["service_area"],
                            labels,
                            confidences
                        )
                    )

                df["image_labels"] = all_labels
                df["image_confidence_avg"] = all_confidences
                df["multimodal_support"] = multimodal_supports
        else:
            df["image_labels"] = ""
            df["image_confidence_avg"] = 0.0
            df["multimodal_support"] = 0.0

        st.caption(
            "Çok modlu destek, görseldeki sorunun şiddetini değil; metinde belirlenen "
            "hizmet alanının ilişkili görsel temas noktasıyla bağlamsal uyumunu gösterir."
        )

        display_cols = [
            "content", "topic", "service_area", "sentiment", "negative_score",
            "image_files", "image_labels", "image_confidence_avg", "multimodal_support"
        ]
        existing_cols = [col for col in display_cols if col in df.columns]
        st.dataframe(df[existing_cols], use_container_width=True)

        # ---------- ADIM 6: KARAR MATRİSİ + CRITIC ----------
        st.subheader("Adım 6 — Karar Matrisinin Oluşturulması ve CRITIC Ağırlıklandırma")

        decision_df = build_decision_matrix(df)

        if len(decision_df) < 2:
            st.error(
                "MCDM analizi için en az iki farklı hizmet alanı gereklidir. "
                "Konu modelleme ayarını veya veri setini kontrol edin."
            )
            st.stop()

        normalized_df = normalize_decision_matrix(decision_df)
        critic_weights = calculate_critic_weights(normalized_df)

        st.markdown("**Başlangıç karar matrisi**")
        st.dataframe(decision_df, use_container_width=True)

        st.markdown("**CRITIC için normalize edilmiş karar matrisi**")
        st.dataframe(normalized_df, use_container_width=True)

        st.markdown("**CRITIC kriter ağırlıkları**")
        st.dataframe(critic_weights, use_container_width=True)

        fig_w, ax_w = plt.subplots()
        chart_w = critic_weights.sort_values("weight", ascending=True)
        ax_w.barh(chart_w["criterion_name"], chart_w["weight"])
        ax_w.set_xlabel("CRITIC ağırlığı")
        ax_w.set_title("Veri Tabanlı Kriter Ağırlıkları")
        st.pyplot(fig_w)
        plt.close(fig_w)

        # ---------- ADIM 7: TOPSIS ----------
        st.subheader("Adım 7 — TOPSIS ile Hizmet İyileştirme Önceliklerinin Belirlenmesi")

        topsis_df = calculate_topsis(decision_df, critic_weights)
        st.dataframe(topsis_df, use_container_width=True)

        if not topsis_df.empty:
            top_row = topsis_df.iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Hizmet Alanı Sayısı", len(topsis_df))
            c2.metric("En Öncelikli Hizmet Alanı", top_row["service_area"])
            c3.metric("TOPSIS Yakınlık Skoru", round(float(top_row["topsis_score"]), 3))

            fig_s, ax_s = plt.subplots()
            score_df = topsis_df.sort_values("topsis_score", ascending=True)
            ax_s.barh(score_df["service_area"], score_df["topsis_score"])
            ax_s.set_xlabel("TOPSIS yakınlık skoru")
            ax_s.set_title("Hizmet İyileştirme Öncelikleri")
            st.pyplot(fig_s)
            plt.close(fig_s)

            st.markdown("**Yönetimsel yorum**")
            st.write(
                f"CRITIC-TOPSIS sonuçlarına göre **{top_row['service_area']}** "
                f"hizmet alanı bu veri setinde en yüksek iyileştirme önceliğine sahiptir. "
                "Sonuç, yaygınlık, olumsuz duygu ve çok modlu destek göstergelerinin "
                "veri tabanlı ağırlıkları birlikte dikkate alınarak elde edilmiştir."
            )

        # ---------- İNDİRME ----------
        st.divider()
        st.subheader("Çıktıları İndir")

        d1, d2, d3 = st.columns(3)

        with d1:
            st.download_button(
                "Analiz Edilmiş Yorumlar",
                data=df.to_csv(index=False).encode("utf-8-sig"),
                file_name="analiz_edilmis_yorumlar.csv",
                mime="text/csv"
            )

        with d2:
            st.download_button(
                "CRITIC Ağırlıkları",
                data=critic_weights.to_csv(index=False).encode("utf-8-sig"),
                file_name="critic_agirliklari.csv",
                mime="text/csv"
            )

        with d3:
            st.download_button(
                "TOPSIS Sonuçları",
                data=topsis_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="topsis_sonuclari.csv",
                mime="text/csv"
            )


# ==================================================
# EKRAN 2: MANUEL YORUM VE GÖRSEL ANALİZİ
# ==================================================

elif sayfa == "2. Manuel Yorum ve Görsel Analizi":

    st.header("2. Manuel Yorum ve Görsel Analizi")
    st.caption(
        "Bu ekran tek bir kayıt üzerinde metin duygu analizi, hizmet alanı eşleştirmesi "
        "ve opsiyonel görsel temas noktası sınıflandırmasını gösterir."
    )

    manual_text = st.text_area(
        "Yolcu yorumunu giriniz",
        height=150,
        placeholder="Örnek: Security queue was very long but the staff were helpful."
    )

    uploaded_image = st.file_uploader(
        "Bir görsel yükleyiniz (opsiyonel)",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if st.button("Manuel Analizi Başlat", type="primary"):

        if not manual_text.strip():
            st.error("Lütfen bir yorum giriniz.")
        else:
            signed_sentiment, negative_score = get_sentiment_scores(manual_text)
            service_area = manual_service_area_mapping(manual_text)

            image_label = "Görsel yok"
            image_confidence = 0.0
            multimodal_support = 0.0

            if uploaded_image is not None:
                image = Image.open(uploaded_image).convert("RGB")
                st.image(image, caption="Yüklenen Görsel", use_container_width=True)

                with st.spinner("Görsel temas noktası sınıflandırılıyor..."):
                    image_label, image_confidence = classify_image(image)

                multimodal_support = service_area_multimodal_support(
                    service_area,
                    [image_label],
                    [image_confidence]
                )

            st.subheader("Manuel Analiz Sonucu")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Hizmet Alanı", service_area)
            c2.metric("İmzalı Duygu", round(signed_sentiment, 3))
            c3.metric("Olumsuzluk Skoru", round(negative_score, 3))
            c4.metric("Çok Modlu Destek", round(multimodal_support, 3))

            st.write(f"**Görsel temas noktası:** {image_label}")
            st.write(f"**Görsel sınıflandırma güveni:** {image_confidence:.3f}")

            st.json({
                "yorum": manual_text,
                "hizmet_alani": service_area,
                "duygu_skoru": signed_sentiment,
                "olumsuz_duygu_skoru": negative_score,
                "gorsel_temas_noktasi": image_label,
                "gorsel_guven_skoru": image_confidence,
                "cok_modlu_destek": multimodal_support
            })
