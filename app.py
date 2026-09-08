
import re
import numpy as np
import pandas as pd
import streamlit as st
import requests
from io import BytesIO
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="TerminAI | Havalimanı Yolcu Terminali Hizmetleri",
    layout="wide"
)

st.title("TerminAI")
st.subheader(
    "Havalimanı Yolcu Terminali Hizmetlerinin İyileştirilmesine Yönelik "
    "Çok Modlu Yapay Zekâ Tabanlı Karar Destek Sistemi"
)
st.caption(
    "UGC → Önişleme → Konu/Hizmet Alanı Keşfi → Duygu Analizi → "
    "Çok Modlu Destek → Karar Matrisi / CRITIC → TOPSIS"
)

st.info(
    "Bu sürüm Render Free (512 MB RAM) için düşük bellekli ön prototiptir. "
    "Transformer/CLIP/BERTopic modelleri web sunucusunda canlı olarak yüklenmez. "
    "Nihai araştırma yönteminde BERTopic ve görüntü sınıflandırma modelleri korunmaktadır; "
    "bu web demosu önceden hesaplanmış görsel etiketlerini kullanabilir."
)

sayfa = st.sidebar.radio(
    "Ekran Seçiniz",
    ["1. Veri Seti Analizi", "2. Manuel Yorum Analizi"]
)

with st.sidebar.expander("Proje metodolojik akışı", expanded=True):
    st.markdown(
        """
        1. **Veri**
        2. **Önişleme**
        3. **Konu modelleme / hizmet alanı keşfi**
        4. **Duygu analizi**
        5. **Görüntü işleme / çok modlu destek**
        6. **Karar matrisi + CRITIC**
        7. **TOPSIS karar modelleme**
        """
    )

EXCEL_URL = (
    "https://raw.githubusercontent.com/bahribarankocak/"
    "airport-dashboard/main/reviews.xlsx"
)

# --------------------------------------------------
# VERİ / ÖNİŞLEME
# --------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_excel_data(url):
    return pd.read_excel(url)

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9çğıöşü\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# --------------------------------------------------
# HİZMET ALANI
# --------------------------------------------------

SERVICE_RULES = {
    "Güvenlik ve pasaport kontrolü": [
        "security", "passport", "screening", "control", "güvenlik", "pasaport"
    ],
    "Bagaj hizmetleri": [
        "baggage", "luggage", "bag", "carousel", "bagaj", "bavul"
    ],
    "Bekleme alanı ve biniş kapısı": [
        "seat", "waiting", "gate", "boarding", "queue", "crowd",
        "bekleme", "koltuk", "kapı", "kuyruk"
    ],
    "Yiyecek-içecek ve perakende": [
        "food", "restaurant", "cafe", "shop", "retail", "price",
        "yemek", "restoran", "kafe", "mağaza"
    ],
    "Tuvalet ve temizlik": [
        "toilet", "restroom", "clean", "dirty", "hygiene",
        "tuvalet", "temiz", "kirli", "hijyen"
    ],
    "Personel hizmetleri": [
        "staff", "employee", "personnel", "rude", "helpful",
        "personel", "çalışan", "yardımcı"
    ],
    "Dijital hizmetler": [
        "wifi", "internet", "charging", "wi-fi", "şarj"
    ],
    "Check-in süreçleri": [
        "check-in", "checkin", "check in"
    ],
}

def service_area_from_text(text):
    t = preprocess_text(text)
    scores = {}
    for area, words in SERVICE_RULES.items():
        scores[area] = sum(1 for w in words if w in t)
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Diğer"

# --------------------------------------------------
# DÜŞÜK BELLEKLİ DUYGU ANALİZİ
# --------------------------------------------------

NEGATIVE_WORDS = {
    "bad","poor","dirty","slow","long","crowded","expensive","rude","delay",
    "delayed","worst","terrible","awful","uncomfortable","problem","queue",
    "broken","difficult","chaotic","unhelpful","late","lost","cold","dark",
    "kötü","kirli","yavaş","uzun","kalabalık","pahalı","kaba","gecikme",
    "rahatsız","sorun","bozuk","zor","kaotik","geç"
}

POSITIVE_WORDS = {
    "good","great","clean","fast","quick","comfortable","helpful","excellent",
    "easy","nice","friendly","efficient","modern","spacious","smooth",
    "iyi","harika","temiz","hızlı","rahat","yardımcı","mükemmel","kolay",
    "güzel","nazik","modern","ferah"
}

def lightweight_sentiment(text):
    tokens = re.findall(r"[a-zA-ZçğıöşüÇĞİÖŞÜ'-]+", str(text).lower())
    if not tokens:
        return 0.0, 0.0
    neg = sum(t in NEGATIVE_WORDS for t in tokens)
    pos = sum(t in POSITIVE_WORDS for t in tokens)
    total = neg + pos
    if total == 0:
        return 0.0, 0.0
    signed = (pos - neg) / total
    negative = neg / total
    return float(signed), float(negative)

# --------------------------------------------------
# HAFİF KONU KEŞFİ (demo amaçlı)
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def lightweight_topic_clusters(texts_tuple, n_clusters):
    texts = list(texts_tuple)
    if len(texts) < 2:
        return [0] * len(texts), {}
    n_clusters = max(2, min(n_clusters, len(texts)))
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_features=1500
    )
    X = vec.fit_transform(texts)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    terms = np.array(vec.get_feature_names_out())
    top_terms = {}
    for i in range(n_clusters):
        order = model.cluster_centers_[i].argsort()[::-1][:7]
        top_terms[i] = terms[order].tolist()
    return labels.tolist(), top_terms

# --------------------------------------------------
# ÇOK MODLU DESTEK
# --------------------------------------------------

IMAGE_TO_SERVICE = {
    "security_area": "Güvenlik ve pasaport kontrolü",
    "waiting_area": "Bekleme alanı ve biniş kapısı",
    "boarding_gate": "Bekleme alanı ve biniş kapısı",
    "baggage_claim": "Bagaj hizmetleri",
    "food_retail_area": "Yiyecek-içecek ve perakende",
    "restroom": "Tuvalet ve temizlik",
    "checkin_area": "Check-in süreçleri",
}

def multimodal_support(service_area, image_label, image_confidence=1.0):
    if pd.isna(image_label) or not str(image_label).strip():
        return 0.0
    match = IMAGE_TO_SERVICE.get(str(image_label).strip())
    if match == service_area:
        try:
            return float(image_confidence)
        except Exception:
            return 1.0
    return 0.0

# --------------------------------------------------
# MCDM
# --------------------------------------------------

CRITERIA = ["prevalence", "negative_sentiment", "multimodal_support"]

def build_decision_matrix(df):
    work = df[df["service_area"] != "Diğer"].copy()
    if work.empty:
        return pd.DataFrame(columns=["service_area"] + CRITERIA)

    total = len(work)
    rows = []
    for area, g in work.groupby("service_area"):
        rows.append({
            "service_area": area,
            "prevalence": len(g) / total,
            "negative_sentiment": float(g["negative_score"].mean()),
            "multimodal_support": float(g["multimodal_support"].mean()),
        })
    return pd.DataFrame(rows)

def minmax_normalize(df):
    out = df[["service_area"]].copy()
    for c in CRITERIA:
        x = df[c].astype(float)
        rng = x.max() - x.min()
        out[c] = 0.0 if rng == 0 else (x - x.min()) / rng
    return out

def critic_weights(normalized_df):
    X = normalized_df[CRITERIA].astype(float)
    std = X.std(ddof=0)
    corr = X.corr().fillna(0.0)
    info = {}
    for c in CRITERIA:
        conflict = sum(1 - corr.loc[c, other] for other in CRITERIA)
        info[c] = float(std[c] * conflict)

    total = sum(info.values())
    if total <= 0:
        weights = {c: 1 / len(CRITERIA) for c in CRITERIA}
    else:
        weights = {c: v / total for c, v in info.items()}

    names = {
        "prevalence": "C1 — Yaygınlık",
        "negative_sentiment": "C2 — Olumsuz Duygu",
        "multimodal_support": "C3 — Çok Modlu Destek",
    }
    return pd.DataFrame({
        "criterion": CRITERIA,
        "criterion_name": [names[c] for c in CRITERIA],
        "weight": [weights[c] for c in CRITERIA],
    })

def topsis(decision_df, weights_df):
    if decision_df.empty:
        return pd.DataFrame()

    X = decision_df[CRITERIA].astype(float).to_numpy()
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    R = X / denom

    wmap = dict(zip(weights_df["criterion"], weights_df["weight"]))
    w = np.array([wmap[c] for c in CRITERIA])
    V = R * w

    ideal = V.max(axis=0)
    anti = V.min(axis=0)
    d_pos = np.sqrt(((V - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((V - anti) ** 2).sum(axis=1))
    score = d_neg / np.where((d_pos + d_neg) == 0, 1, d_pos + d_neg)

    out = decision_df.copy()
    out["topsis_score"] = score
    out["rank"] = out["topsis_score"].rank(ascending=False, method="min").astype(int)
    return out.sort_values(["rank", "topsis_score"], ascending=[True, False]).reset_index(drop=True)

# --------------------------------------------------
# EKRAN 1
# --------------------------------------------------

if sayfa == "1. Veri Seti Analizi":
    st.header("1. Veri Seti Analizi")

    data_source = st.radio(
        "Veri kaynağı",
        ["GitHub demo veri seti", "Excel dosyası yükle"],
        horizontal=True
    )

    try:
        if data_source == "GitHub demo veri seti":
            with st.spinner("Veri seti yükleniyor..."):
                df = load_excel_data(EXCEL_URL)
        else:
            uploaded = st.file_uploader("Excel dosyası", type=["xlsx", "xls"])
            if uploaded is None:
                st.stop()
            df = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Veri seti yüklenemedi: {e}")
        st.stop()

    if "content" not in df.columns:
        st.error("Veri setinde 'content' sütunu bulunmalıdır.")
        st.stop()

    st.success(f"{len(df)} yorum yüklendi.")

    if st.button("Analizi Başlat", type="primary"):
        with st.spinner("Düşük bellekli analiz yürütülüyor..."):
            df = df.copy()
            df["topic_text"] = df["content"].fillna("").map(preprocess_text)

            n_clusters = min(6, max(2, int(np.sqrt(max(len(df), 2)))))
            cluster_labels, cluster_terms = lightweight_topic_clusters(
                tuple(df["topic_text"].tolist()),
                n_clusters
            )
            df["topic"] = cluster_labels
            df["service_area"] = df["content"].map(service_area_from_text)

            sentiments = df["content"].map(lightweight_sentiment)
            df["sentiment"] = sentiments.map(lambda x: x[0])
            df["negative_score"] = sentiments.map(lambda x: x[1])

            # Önceden hesaplanmış görsel sınıflandırma varsa kullan.
            if "image_label" in df.columns:
                if "image_confidence" not in df.columns:
                    df["image_confidence"] = 1.0
                df["multimodal_support"] = df.apply(
                    lambda r: multimodal_support(
                        r["service_area"], r["image_label"], r["image_confidence"]
                    ),
                    axis=1
                )
            elif "image_labels" in df.columns:
                # İlk etiketi kullanır; varsa confidence ortalaması.
                def first_label(v):
                    if pd.isna(v):
                        return ""
                    return str(v).split("|")[0].strip()
                df["_image_label"] = df["image_labels"].map(first_label)
                conf_col = "image_confidence_avg" if "image_confidence_avg" in df.columns else None
                df["multimodal_support"] = df.apply(
                    lambda r: multimodal_support(
                        r["service_area"],
                        r["_image_label"],
                        r[conf_col] if conf_col else 1.0
                    ),
                    axis=1
                )
            else:
                df["multimodal_support"] = 0.0

        st.subheader("Adım 3 — Konu / Hizmet Alanı Keşfi")
        topic_rows = []
        for k, terms in cluster_terms.items():
            topic_rows.append({"topic": k, "top_terms": ", ".join(terms)})
        st.dataframe(pd.DataFrame(topic_rows), use_container_width=True)

        st.subheader("Adım 4–5 — Duygu ve Çok Modlu Destek")
        cols = [
            c for c in [
                "content", "topic", "service_area", "sentiment",
                "negative_score", "image_label", "image_labels",
                "multimodal_support"
            ] if c in df.columns
        ]
        st.dataframe(df[cols], use_container_width=True)

        if float(df["multimodal_support"].sum()) == 0:
            st.warning(
                "Bu veri dosyasında önceden hesaplanmış görsel sınıf etiketi bulunmadığı "
                "için C3 şu anda 0'dır. Ücretsiz 512 MB sürümünde CLIP canlı yüklenmez. "
                "Görsel etiketleri offline hesaplanıp Excel'e 'image_label' ve "
                "'image_confidence' sütunları olarak eklendiğinde C3 otomatik çalışır."
            )

        st.subheader("Adım 6 — Karar Matrisi ve CRITIC")
        decision = build_decision_matrix(df)
        if len(decision) < 2:
            st.error("MCDM analizi için en az iki farklı hizmet alanı gereklidir.")
            st.stop()

        normalized = minmax_normalize(decision)
        weights = critic_weights(normalized)

        st.markdown("**Başlangıç karar matrisi**")
        st.dataframe(decision, use_container_width=True)
        st.markdown("**Normalize edilmiş karar matrisi**")
        st.dataframe(normalized, use_container_width=True)
        st.markdown("**CRITIC kriter ağırlıkları**")
        st.dataframe(weights, use_container_width=True)

        st.subheader("Adım 7 — TOPSIS ile İyileştirme Öncelikleri")
        ranking = topsis(decision, weights)
        st.dataframe(ranking, use_container_width=True)

        if not ranking.empty:
            top = ranking.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Hizmet Alanı Sayısı", len(ranking))
            c2.metric("En Öncelikli Alan", top["service_area"])
            c3.metric("TOPSIS Skoru", f"{top['topsis_score']:.3f}")

        st.download_button(
            "TOPSIS Sonuçlarını İndir",
            ranking.to_csv(index=False).encode("utf-8-sig"),
            "topsis_sonuclari.csv",
            "text/csv"
        )

# --------------------------------------------------
# EKRAN 2
# --------------------------------------------------

else:
    st.header("2. Manuel Yorum Analizi")
    st.caption(
        "Bu düşük bellekli sürümde metin analizi canlıdır. "
        "Canlı görüntü sınıflandırması 512 MB sınırı nedeniyle devre dışıdır."
    )

    text = st.text_area(
        "Yolcu yorumunu giriniz",
        height=150,
        placeholder="Örnek: Security queue was very long and the area was crowded."
    )

    if st.button("Yorumu Analiz Et", type="primary"):
        if not text.strip():
            st.error("Lütfen bir yorum giriniz.")
        else:
            area = service_area_from_text(text)
            signed, negative = lightweight_sentiment(text)
            c1, c2, c3 = st.columns(3)
            c1.metric("Hizmet Alanı", area)
            c2.metric("Duygu Skoru", f"{signed:.2f}")
            c3.metric("Olumsuz Duygu", f"{negative:.2f}")
