import streamlit as st
import pandas as pd
import numpy as np
import re
 
st.set_page_config(page_title="AI Havalimanı Karar Destek", layout="wide")
 
@st.cache_resource(show_spinner="Embedding modeli yükleniyor...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
 
@st.cache_resource(show_spinner="Duygu modeli yükleniyor...")
def load_sentiment_model():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        top_k=1
    )
 
def split_sentences(text):
    parts = re.split(r'[.!?]|\bbut\b|\bhowever\b|\bfakat\b|\bama\b|\bancak\b', text, flags=re.IGNORECASE)
    return [p.strip() for p in parts if len(p.strip()) > 8]
 
def run_bertopic(sentences, embedding_model, n_topics=5):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
 
    if len(sentences) < 5:
        return None, None, None
 
    umap_model = UMAP(
        n_neighbors=min(5, len(sentences) - 1),
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=42
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=max(2, len(sentences) // (n_topics + 1)),
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True
    )
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=n_topics,
        verbose=False
    )
    topics, probs = topic_model.fit_transform(sentences)
    return topic_model, topics, probs
 
def get_topic_label(topic_model, topic_id):
    if topic_id == -1:
        return "Gurultu / Siniflandirilamadi"
    words = topic_model.get_topic(topic_id)
    if not words:
        return f"Konu {topic_id}"
    top_words = [w for w, _ in words[:4]]
    return f"Konu {topic_id}: {', '.join(top_words)}"
 
def get_sentiment(texts, sentiment_pipeline):
    results = []
    for text in texts:
        text = text[:512]
        try:
            out = sentiment_pipeline(text)[0][0]
            label = out["label"].lower()
            score = out["score"]
            if label in ["positive", "pos"]:
                results.append(("Pozitif", round(score, 3)))
            elif label in ["negative", "neg"]:
                results.append(("Negatif", round(-score, 3)))
            else:
                results.append(("Notr", 0.0))
        except Exception:
            results.append(("Notr", 0.0))
    return results
 
def run_pipeline(entries, embedding_model, sentiment_pipeline, n_topics):
    all_sentences = []
    sentence_meta = []
 
    for entry in entries:
        sents = split_sentences(entry["text"])
        if not sents:
            sents = [entry["text"]]
        for s in sents:
            all_sentences.append(s)
            sentence_meta.append({"airport": entry["airport"]})
 
    if len(all_sentences) < 5:
        st.warning("En az 5 cumle gerekiyor.")
        return None, None, None
 
    with st.spinner("BERTopic konu modeli egitiliyor..."):
        topic_model, topics, probs = run_bertopic(all_sentences, embedding_model, n_topics)
 
    if topic_model is None:
        st.warning("Yeterli veri yok.")
        return None, None, None
 
    with st.spinner("Duygu analizi yapiliyor..."):
        sentiments = get_sentiment(all_sentences, sentiment_pipeline)
 
    rows = []
    for i, sent in enumerate(all_sentences):
        topic_id = topics[i]
        topic_label = get_topic_label(topic_model, topic_id)
        duygu, skor = sentiments[i]
        rows.append({
            "Havalimani": sentence_meta[i]["airport"],
            "Konu_ID": topic_id,
            "Konu": topic_label,
            "Cumle": sent,
            "Duygu": duygu,
            "Skor": skor,
        })
 
    df = pd.DataFrame(rows)
    return df, topic_model, all_sentences
 
def aggregate(df):
    agg = df[df["Konu_ID"] != -1].groupby("Konu").agg(
        Cumle_Sayisi=("Skor", "count"),
        Ort_Skor=("Skor", "mean"),
        Pozitif=("Duygu", lambda x: (x == "Pozitif").sum()),
        Negatif=("Duygu", lambda x: (x == "Negatif").sum()),
        Notr=("Duygu", lambda x: (x == "Notr").sum()),
    ).reset_index()
    agg["Ort_Skor"] = agg["Ort_Skor"].round(3)
    agg["Baskin_Duygu"] = agg["Ort_Skor"].apply(
        lambda s: "Pozitif" if s > 0.05 else ("Negatif" if s < -0.05 else "Notr")
    )
    return agg.sort_values("Ort_Skor")
 
def sentiment_color(label):
    return {"Pozitif": "#2e7d32", "Negatif": "#c62828", "Notr": "#e65100"}.get(label, "#888")
 
def render_topic_bar(row):
    skor = row["Ort_Skor"]
    color = sentiment_color(row["Baskin_Duygu"])
    bar_w = int(abs(skor) * 100)
    st.markdown(f"""
    <div style="margin-bottom:10px; padding:10px 14px;
                border:0.5px solid #ddd; border-radius:8px; background:#fafafa;">
        <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <b>{row['Konu']}</b>
            <span style="color:{color}; font-weight:500">{row['Baskin_Duygu']} &nbsp; {skor:+.3f}</span>
        </div>
        <div style="background:#eee; border-radius:4px; height:8px; width:100%">
            <div style="background:{color}; width:{bar_w}%; height:8px; border-radius:4px;"></div>
        </div>
        <small style="color:#888">
            {int(row['Cumle_Sayisi'])} cumle &nbsp;|&nbsp;
            {int(row['Pozitif'])} pozitif &nbsp;
            {int(row['Negatif'])} negatif &nbsp;
            {int(row['Notr'])} notr
        </small>
    </div>
    """, unsafe_allow_html=True)
 
vision_mock = {
    "Lounge / Bekleme Alani": ["Koltuk Dolulugu: %85", "Atik/Kirlilik: Saptandi", "Aydinlatma: Yeterli"],
    "Check-in / Kontuvar": ["Kuyruk Uzunlugu: >10m", "Personel Sayisi: 2", "Bekleme Suresi: Yuksek"],
    "Yeme-Icme Alani": ["Masa Temizligi: Dusuk", "Gida Cesitliligi: Orta", "Yogunluk: Orta"],
    "Guvenlik Gecisi": ["Arama Noktasi: 3 Acik", "Akis Hizi: Yavas", "Gerginlik Skoru: Orta"]
}
 
st.title("Havalimani Pazarlama Karar Destek Sistemi")
st.markdown("### Multimodal Analiz Dashboard - BERTopic + XLM-RoBERTa")
 
st.sidebar.header("Analiz Parametreleri")
analysis_mode = st.sidebar.radio("Analiz Modu:", ["Tekli Yorum", "Toplu Yorum (BERTopic)"])
location_type = st.sidebar.selectbox(
    "Havalimani Bolgesi:",
    ["Lounge / Bekleme Alani", "Check-in / Kontuvar", "Yeme-Icme Alani", "Guvenlik Gecisi"]
)
 
if analysis_mode == "Tekli Yorum":
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Veri Giris Katmani")
        uploaded_file = st.file_uploader("Fotograf Yukle", type=["jpg", "png", "jpeg"])
        user_review = st.text_area(
            "Musteri Yorumu (TR veya EN):", height=150,
            placeholder="The food was great but the lounge was incredibly crowded and dirty."
        )
    with col2:
        st.subheader("Analiz Merkezi")
        if uploaded_file and user_review.strip():
            st.image(uploaded_file, caption=f"Bolge: {location_type}", use_container_width=True)
            st.write("---")
            st.markdown("**Goruntu Kriterleri:**")
            items = vision_mock.get(location_type, [])
            if items:
                cols = st.columns(len(items))
                for i, item in enumerate(items):
                    cols[i].info(item)
            st.write("---")
            st.markdown("**Duygu Analizi (XLM-RoBERTa):**")
            sentiment_pipeline = load_sentiment_model()
            sents = split_sentences(user_review) or [user_review]
            results = get_sentiment(sents, sentiment_pipeline)
            for sent, (label, score) in zip(sents, results):
                color = sentiment_color(label)
                st.markdown(f"""
                <div style="border-left:4px solid {color}; padding:8px 12px; margin-bottom:8px; border-radius:4px;">
                    <b style="color:{color}">{label} ({score:+.3f})</b><br>
                    <small style="color:#555">{sent}</small>
                </div>""", unsafe_allow_html=True)
            st.caption(f"MCDM modelindeki '{location_type}' agirligi guncellendi.")
        else:
            st.info("Fotograf yukleyin ve yorum girin.")
 
else:
    st.subheader("Toplu Yorum Analizi - BERTopic Pipeline")
    st.info(
        "Yorumlar cumlere ayrilir -> embedding -> UMAP boyut indirgeme -> "
        "HDBSCAN kumeleme -> BERTopic konu etiketleri -> XLM-RoBERTa duygu skoru"
    )
 
    n_topics = st.sidebar.slider("Hedef Konu Sayisi (k):", min_value=2, max_value=10, value=5)
 
    default_reviews = """IST | Security queue was incredibly long, waited 45 minutes. But the lounge was fantastic.
SAW | Check-in staff were rude and unhelpful. Baggage took forever to arrive.
AMS | The lounge was clean and quiet, great place to relax before the flight.
IST | Guvenlik kontrolu cok hizliydi, personel guler yuzluydu.
LHR | Wi-Fi didn't work at all. Shops were overpriced but the food was decent.
SAW | Bagaj bekleme alani cok kalabalıkti, valizim 1 saat sonra cikti.
AMS | Passport control was smooth and fast. Staff very professional.
IST | Terminalin temizligi berbatti, tuvaletler igrencti. Kalkis gecikti.
LHR | The seating in the departure lounge was comfortable but Wi-Fi was unreliable.
SAW | Check-in sureci hizliydi ama guvenlik noktasinda uzun kuyruk vardi.
IST | The food court had great variety but prices were extremely high.
AMS | Lost my baggage, no one at the counter could help me properly.
LHR | Lounge staff were incredibly welcoming and the seats were very comfortable.
IST | Guvenlik noktasinda personel cok kabaydı, hic yardimci olmadılar.
SAW | Wi-Fi baglantisi surekli koptu, calismam cok zorlasti."""
 
    bulk_input = st.text_area("Yorumlar (Havalimani | Yorum):", value=default_reviews, height=300)
 
    if st.button("BERTopic Analizini Baslat", type="primary"):
        entries = []
        for line in bulk_input.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            sep = "|" if "|" in line else (";" if ";" in line else None)
            if sep:
                idx = line.index(sep)
                a, t = line[:idx].strip(), line[idx+1:].strip()
                if a and t:
                    entries.append({"airport": a, "text": t})
            else:
                entries.append({"airport": "-", "text": line})
 
        if not entries:
            st.warning("Yorum bulunamadi.")
        else:
            embedding_model = load_embedding_model()
            sentiment_pipeline = load_sentiment_model()
            df, topic_model, all_sentences = run_pipeline(entries, embedding_model, sentiment_pipeline, n_topics)
 
            if df is not None:
                agg = aggregate(df)
                st.write("---")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Yorum", len(entries))
                c2.metric("Cumle", len(df))
                c3.metric("Konu", df[df["Konu_ID"] != -1]["Konu_ID"].nunique())
                c4.metric("Siniflandirilamadi", int((df["Konu_ID"] == -1).sum()))
 
                st.write("---")
                st.markdown("**Konu Bazinda Duygu Profili:**")
                for _, row in agg.iterrows():
                    render_topic_bar(row)
 
                st.write("---")
                st.markdown("**Konu Anahtar Kelimeleri (BERTopic):**")
                topic_ids = sorted(df[df["Konu_ID"] != -1]["Konu_ID"].unique())
                if topic_ids:
                    cols = st.columns(min(len(topic_ids), 3))
                    for i, tid in enumerate(topic_ids):
                        words = topic_model.get_topic(tid)
                        if words:
                            top = [f"{w} ({round(s,3)})" for w, s in words[:6]]
                            cols[i % 3].markdown(
                                f"**Konu {tid}**\n\n" + "\n\n".join(f"- {w}" for w in top)
                            )
 
                st.write("---")
                st.markdown("**Ham Analiz Tablosu:**")
                st.dataframe(df, use_container_width=True)
 
                csv = df.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "CSV olarak indir",
                    data=csv,
                    file_name="bertopic_absa.csv",
                    mime="text/csv"
                )
                st.caption(
                    "BERTopic konu etiketleri ve duygu skorlari MCDM modeline girdi olarak kullanilabilir."
                )
 
