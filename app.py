"""
NLP Project 2 - Application Streamlit
Analyse d'avis clients assurance francaise
ESILV DIA4 - 2025/2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(page_title="Analyse Avis Assurance", layout="wide", page_icon="📊")

st.markdown("""
<style>
    /* Fond principal */
    .stApp { background-color: #0f1724; color: #e2e8f0; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a2744 0%, #0f1724 100%);
        border-right: 1px solid #2d4a7a;
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; }

    /* Titres */
    h1 { color: #60a5fa !important; font-weight: 700 !important; border-bottom: 2px solid #2d4a7a; padding-bottom: 0.4rem; }
    h2, h3 { color: #93c5fd !important; }

    /* Métriques */
    [data-testid="stMetric"] {
        background: #1e2d4a;
        border: 1px solid #2d4a7a;
        border-radius: 10px;
        padding: 12px 16px !important;
    }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { color: #60a5fa !important; font-size: 1.5rem !important; }

    /* Boutons */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        border: none !important;
        border-radius: 8px !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
        box-shadow: 0 4px 12px rgba(37,99,235,0.4) !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
        box-shadow: 0 6px 16px rgba(37,99,235,0.5) !important;
        transform: translateY(-1px) !important;
    }

    /* Inputs */
    .stTextInput input, .stTextArea textarea {
        background: #1e2d4a !important;
        border: 1px solid #2d4a7a !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 2px rgba(96,165,250,0.2) !important;
    }

    /* Selectbox / slider */
    .stSelectbox select, [data-baseweb="select"] {
        background: #1e2d4a !important;
        border-color: #2d4a7a !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stSlider"] { accent-color: #3b82f6; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #2d4a7a; border-radius: 8px; }

    /* Info / warning / error / success */
    [data-testid="stAlert"] { border-radius: 10px !important; }
    .stSuccess { background: #052e16 !important; border-color: #16a34a !important; color: #86efac !important; }
    .stError { background: #2d0a0a !important; border-color: #dc2626 !important; color: #fca5a5 !important; }
    .stWarning { background: #1c1500 !important; border-color: #ca8a04 !important; color: #fde68a !important; }
    .stInfo { background: #0c1d3a !important; border-color: #2563eb !important; color: #93c5fd !important; }

    /* Spinner */
    .stSpinner > div { border-top-color: #3b82f6 !important; }
</style>
""", unsafe_allow_html=True)


# Mots vides francais
STOPWORDS_FR = set(stopwords.words('french'))
STOPWORDS_FR.update(['très', 'bien', 'plus', 'tout', 'ça', 'car', 'comme',
                     'cela', 'non', 'après', 'dès', 'lors', 'assurance',
                     'avoir', 'être'])


def nettoyer_texte(texte):
    """Nettoie un avis : minuscules, sans ponctuation, sans mots vides."""
    if not isinstance(texte, str) or len(texte.strip()) == 0:
        return ''
    texte = texte.lower()
    texte = re.sub(r'http\S+|www\.\S+', '', texte)
    texte = re.sub(r'[^a-zàâäéèêëîïôùûüç\s]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    try:
        tokens = word_tokenize(texte, language='french')
    except Exception:
        tokens = texte.split()
    tokens = [t for t in tokens if t not in STOPWORDS_FR and len(t) > 2]
    return ' '.join(tokens)


def note_vers_sentiment(note):
    """Convertit une note en sentiment."""
    if note >= 4:
        return 'positif'
    elif note == 3:
        return 'neutre'
    else:
        return 'negatif'


def resumer_avis(liste_avis, nb_phrases=3):
    """Genere un resume extractif par similarite TF-IDF cosinus."""
    phrases = []
    for avis in liste_avis:
        if isinstance(avis, str) and len(avis.strip()) > 20:
            for p in re.split(r'[.!?]+', avis):
                p = p.strip()
                if len(p) > 20:
                    phrases.append(p)
    if len(phrases) <= nb_phrases:
        return ' '.join(phrases)
    try:
        vec = TfidfVectorizer(max_features=300, stop_words=list(STOPWORDS_FR))
        mat = vec.fit_transform(phrases)
        sim = cosine_similarity(mat, mat)
        scores = sim.sum(axis=1)
        top_idx = sorted(scores.argsort()[-nb_phrases:].tolist())
        return ' '.join([phrases[i] for i in top_idx])
    except Exception:
        return ' '.join(phrases[:nb_phrases])


def qa_extractif(question, contexte):
    """Trouve la phrase du contexte la plus proche de la question."""
    phrases = [p.strip() for p in re.split(r'[.!?]+', contexte) if len(p.strip()) > 10]
    if not phrases:
        return contexte[:300]
    try:
        docs = phrases + [question]
        vec = TfidfVectorizer(max_features=500)
        mat = vec.fit_transform(docs)
        sims = cosine_similarity(mat[-1], mat[:-1]).flatten()
        best = sims.argmax()
        return phrases[best]
    except Exception:
        return phrases[0]


@st.cache_data
def charger_donnees():
    """Charge le dataset depuis le CSV ou les fichiers xlsx."""
    if os.path.exists('avis_clients_clean.csv'):
        df = pd.read_csv('avis_clients_clean.csv', encoding='utf-8-sig')
        df['note'] = pd.to_numeric(df['note'], errors='coerce').fillna(3)
        df['sentiment'] = df['note'].apply(note_vers_sentiment)
        if 'avis_clean' not in df.columns:
            df['avis_clean'] = df['avis'].apply(nettoyer_texte)
        return df
    all_files = glob.glob('*.xlsx')
    if all_files:
        dfs = [pd.read_excel(f) for f in sorted(all_files)]
        df = pd.concat(dfs, ignore_index=True)
        df['note'] = pd.to_numeric(df['note'], errors='coerce').fillna(3)
        df['sentiment'] = df['note'].apply(note_vers_sentiment)
        df['avis_clean'] = df['avis'].apply(nettoyer_texte)
        return df
    st.error("Fichier avis_clients_clean.csv introuvable.")
    st.stop()


@st.cache_resource
def entrainer_modele(_df):
    """Entraine un pipeline TF-IDF + Regression Logistique."""
    textes = _df['avis_clean'].fillna('').tolist()
    labels = _df['sentiment'].tolist()
    valides = [(t, l) for t, l in zip(textes, labels) if len(t.strip()) > 5]
    textes_v, labels_v = zip(*valides)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 2), sublinear_tf=True)),
        ('clf', LogisticRegression(max_iter=500, C=1.0, solver='lbfgs'))
    ])
    pipeline.fit(textes_v, labels_v)
    return pipeline


@st.cache_resource
def construire_index(_df):
    """Construit l'index BM25 et la matrice TF-IDF pour la recherche."""
    corpus = _df['avis_clean'].fillna('').tolist()
    bm25 = BM25Okapi([t.split() for t in corpus])
    tfidf = TfidfVectorizer(max_features=10000, min_df=1)
    matrice = tfidf.fit_transform(corpus)
    return bm25, corpus, tfidf, matrice


# Chargement des donnees et des modeles
df = charger_donnees()
modele = entrainer_modele(df)
bm25, corpus, tfidf_search, matrice_search = construire_index(df)

# Couleurs par sentiment (thème sombre)
COULEURS = {'positif': '#052e16', 'neutre': '#1c1004', 'negatif': '#2d0a0a'}
BORDURES = {'positif': '#16a34a', 'neutre': '#ca8a04', 'negatif': '#dc2626'}
TEXTES  = {'positif': '#86efac', 'neutre':  '#fde68a', 'negatif': '#fca5a5'}

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "Accueil",
    "Prediction",
    "Summary",
    "Explanation",
    "Information Retrieval",
    "RAG",
    "QA"
])
st.sidebar.markdown("Nassim LOUDIYI & Paul-Adrien LU-YEN-TUNG  \nESILV DIA4 - 2025/2026")


# Page 0 : Accueil
if page == "Accueil":
    st.title("Analyse d'avis clients — Assurance française")
    st.markdown(
        "Ce projet applique des techniques de **traitement automatique du langage naturel (NLP)** "
        "à un corpus d'avis clients issus de compagnies d'assurance françaises. "
        "L'objectif est d'analyser, classifier et explorer ces avis à travers plusieurs modules : "
        "prédiction de sentiment, résumé automatique, recherche d'information et question-réponse. "
        "Le modèle de classification repose sur un pipeline **TF-IDF + Régression Logistique**."
    )

    st.markdown("---")

    # Stats du dataset
    st.markdown("### Dataset")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avis clients", "34 388")
    col2.metric("Assureurs", "56")
    col3.metric("Classes de sentiment", "3")
    col4.metric("Notes", "1 à 5 / 5")

    st.markdown(
        "<p style='color:#94a3b8;font-size:.9rem;'>Nassim LOUDIYI &amp; Paul-Adrien LU-YEN-TUNG &nbsp;·&nbsp; ESILV DIA4 — 2025/2026</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # Aperçu des pages
    st.markdown("### Pages disponibles")

    PAGES_INFO = [
        ("Prediction",            "Saisissez un avis client : le modèle prédit son sentiment (positif / neutre / négatif) et estime la note associée."),
        ("Summary",               "Sélectionnez un assureur et obtenez un résumé extractif automatique de ses avis, filtrable par sentiment."),
        ("Explanation",           "Visualisez les mots qui ont le plus influencé la prédiction, via les coefficients de la régression logistique."),
        ("Information Retrieval", "Recherchez des avis pertinents par mots-clés grâce à BM25 ou à la similarité cosinus TF-IDF."),
        ("RAG",                   "Retrieval-Augmented Generation : les avis les plus proches d'une requête sont récupérés, puis résumés automatiquement."),
        ("QA",                    "Posez une question en langage naturel : le système retrouve les avis sources et en extrait la phrase réponse la plus pertinente."),
    ]

    for nom, desc in PAGES_INFO:
        st.markdown(f"""
        <div style="background:#1e2d4a;border-left:4px solid #3b82f6;
                    padding:14px 18px;border-radius:8px;margin:8px 0;">
            <span style="color:#60a5fa;font-weight:700;font-size:1rem;">{nom}</span>
            <p style="margin:.3rem 0 0;color:#cbd5e1;line-height:1.5;">{desc}</p>
        </div>""", unsafe_allow_html=True)


# Page 1 : Prediction
elif page == "Prediction":
    st.title("Prediction de sentiment")
    st.write("Entrez un avis client. Le modele predit le sentiment et la note estimee.")

    avis_saisi = st.text_area("Avis client :", height=150,
                               placeholder="Ex: Le service client a ete tres reactif.")

    if st.button("Analyser", type="primary"):
        if not avis_saisi.strip():
            st.warning("Veuillez entrer un avis.")
        else:
            avis_clean = nettoyer_texte(avis_saisi)
            if len(avis_clean) < 3:
                avis_clean = avis_saisi

            # Prediction
            probas = modele.predict_proba([avis_clean])[0]
            classes = modele.classes_
            sentiment_predit = classes[probas.argmax()]
            confiance = probas.max()
            note_estimee = {'positif': '4-5 / 5', 'neutre': '3 / 5', 'negatif': '1-2 / 5'}

            # Affichage du resultat
            if sentiment_predit == 'positif':
                st.success(f"Sentiment : **{sentiment_predit.upper()}** — Confiance : {confiance:.1%}")
            elif sentiment_predit == 'negatif':
                st.error(f"Sentiment : **{sentiment_predit.upper()}** — Confiance : {confiance:.1%}")
            else:
                st.warning(f"Sentiment : **{sentiment_predit.upper()}** — Confiance : {confiance:.1%}")

            col1, col2, col3 = st.columns(3)
            col1.metric("Sentiment", sentiment_predit.capitalize())
            col2.metric("Note estimee", note_estimee[sentiment_predit])
            col3.metric("Confiance", f"{confiance:.1%}")

            # Probabilites par classe
            st.markdown("**Probabilites par classe :**")
            for cls, prob in sorted(zip(classes, probas), key=lambda x: -x[1]):
                couleur = BORDURES.get(cls, '#3b82f6')
                texte_c = TEXTES.get(cls, '#e2e8f0')
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:12px;margin:6px 0;">
                  <span style="width:75px;font-weight:600;color:{texte_c};">{cls.capitalize()}</span>
                  <div style="flex:1;background:#1e2d4a;border-radius:6px;height:22px;overflow:hidden;">
                    <div style="background:{couleur};width:{prob*100:.1f}%;height:100%;border-radius:6px;
                                box-shadow:0 0 8px {couleur}80;transition:width .4s;"></div>
                  </div>
                  <span style="width:55px;color:#94a3b8;text-align:right;">{prob:.1%}</span>
                </div>""", unsafe_allow_html=True)


# Page 2 : Summary
elif page == "Summary":
    st.title("Resume par assureur")
    st.write("Resume automatique des avis d'un assureur par similarite TF-IDF.")

    assureurs = sorted(df['assureur'].dropna().unique().tolist())
    col1, col2, col3 = st.columns(3)
    with col1:
        assureur_choisi = st.selectbox("Assureur :", assureurs)
    with col2:
        filtre = st.radio("Avis :", ["Tous", "positif", "neutre", "negatif"], horizontal=True)
    with col3:
        nb_phrases = st.slider("Phrases dans le resume", 2, 6, 3)

    df_filtre = df[df['assureur'] == assureur_choisi].copy()
    if filtre != "Tous":
        df_filtre = df_filtre[df_filtre['sentiment'] == filtre]

    # Indicateurs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre d'avis", len(df_filtre))
    col2.metric("Note moyenne", f"{df_filtre['note'].mean():.2f}/5")
    col3.metric("% Positifs", f"{(df_filtre['sentiment']=='positif').mean()*100:.1f}%")
    col4.metric("% Negatifs", f"{(df_filtre['sentiment']=='negatif').mean()*100:.1f}%")

    # Distribution des sentiments
    col_a, col_b = st.columns([1, 2])
    with col_a:
        st.write("**Distribution :**")
        st.dataframe(df_filtre['sentiment'].value_counts().rename("Nombre").to_frame())
    with col_b:
        st.bar_chart(df_filtre['sentiment'].value_counts())

    # Resume extractif
    st.markdown("**Resume automatique :**")
    avis_textes = df_filtre['avis'].dropna().tolist()
    if avis_textes:
        with st.spinner("Generation du resume..."):
            resume = resumer_avis(avis_textes, nb_phrases)
        st.info(resume if resume.strip() else "Pas assez d'avis pour ce filtre.")
    else:
        st.warning("Aucun avis disponible.")

    # Exemples d'avis
    st.write("**Exemples d'avis :**")
    st.dataframe(df_filtre[['avis', 'note', 'sentiment']].head(8).reset_index(drop=True))


# Page 3 : Explanation
elif page == "Explanation":
    st.title("Explication des predictions")
    st.write("Le modele TF-IDF + Regression Logistique explique sa prediction mot par mot.")

    avis_saisi = st.text_area("Avis client :", height=150,
                               placeholder="Ex: Remboursement refuse, service client inexistant.")

    if st.button("Expliquer", type="primary"):
        if not avis_saisi.strip():
            st.warning("Veuillez entrer un avis.")
        else:
            avis_clean = nettoyer_texte(avis_saisi)
            if len(avis_clean) < 3:
                avis_clean = avis_saisi

            # Prediction
            probas = modele.predict_proba([avis_clean])[0]
            classes = modele.classes_
            sentiment_predit = classes[probas.argmax()]
            confiance = probas.max()

            if sentiment_predit == 'positif':
                st.success(f"Sentiment predit : **{sentiment_predit.upper()}** ({confiance:.1%})")
            elif sentiment_predit == 'negatif':
                st.error(f"Sentiment predit : **{sentiment_predit.upper()}** ({confiance:.1%})")
            else:
                st.warning(f"Sentiment predit : **{sentiment_predit.upper()}** ({confiance:.1%})")

            # Mots influents via coefficients de la regression logistique
            st.markdown("**Mots les plus influents pour cette prediction :**")
            tfidf_fitted = modele.named_steps['tfidf']
            clf_fitted = modele.named_steps['clf']
            feature_names = tfidf_fitted.get_feature_names_out()
            vecteur = tfidf_fitted.transform([avis_clean])
            features_presentes = vecteur.nonzero()[1]

            if len(features_presentes) == 0:
                st.info("Aucun mot reconnu apres nettoyage.")
            else:
                idx_classe = list(classes).index(sentiment_predit)
                scores_mots = [
                    (feature_names[fi], float(clf_fitted.coef_[idx_classe][fi]))
                    for fi in features_presentes
                ]
                scores_mots.sort(key=lambda x: -abs(x[1]))

                positifs = [(m, s) for m, s in scores_mots if s > 0][:8]
                negatifs = [(m, s) for m, s in scores_mots if s < 0][:8]

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Pour '{sentiment_predit}' :**")
                    max_p = max((abs(s) for _, s in positifs), default=1)
                    for mot, score in positifs:
                        largeur = int(abs(score) / max_p * 100)
                        st.markdown(f"""
                        <div style="margin:5px 0;">
                          <code style="background:#052e16;color:#86efac;padding:3px 8px;border-radius:5px;border:1px solid #16a34a;">{mot}</code>
                          <span style="color:#16a34a;margin-left:8px;font-weight:600;">+{score:.3f}</span>
                          <div style="background:#1e2d4a;height:5px;border-radius:3px;margin-top:4px;">
                            <div style="background:#16a34a;height:5px;width:{largeur}%;border-radius:3px;box-shadow:0 0 6px #16a34a80;"></div>
                          </div>
                        </div>""", unsafe_allow_html=True)
                with col2:
                    st.write("**Contre ce sentiment :**")
                    max_n = max((abs(s) for _, s in negatifs), default=1)
                    for mot, score in negatifs:
                        largeur = int(abs(score) / max_n * 100)
                        st.markdown(f"""
                        <div style="margin:5px 0;">
                          <code style="background:#2d0a0a;color:#fca5a5;padding:3px 8px;border-radius:5px;border:1px solid #dc2626;">{mot}</code>
                          <span style="color:#dc2626;margin-left:8px;font-weight:600;">{score:.3f}</span>
                          <div style="background:#1e2d4a;height:5px;border-radius:3px;margin-top:4px;">
                            <div style="background:#dc2626;height:5px;width:{largeur}%;border-radius:3px;box-shadow:0 0 6px #dc262680;"></div>
                          </div>
                        </div>""", unsafe_allow_html=True)

                # Tableau complet
                st.write("**Tableau des coefficients :**")
                df_coef = pd.DataFrame(scores_mots[:12], columns=['Mot', 'Coefficient'])
                df_coef['Sens'] = df_coef['Coefficient'].apply(
                    lambda x: f"Pour {sentiment_predit}" if x > 0 else "Contre"
                )
                st.dataframe(df_coef.style.background_gradient(
                    subset=['Coefficient'], cmap='RdYlGn', vmin=-1, vmax=1
                ), use_container_width=True)


# Page 4 : Information Retrieval
elif page == "Information Retrieval":
    st.title("Recherche d'avis")
    st.write("Recherche par BM25 ou similarite TF-IDF cosinus.")

    requete = st.text_input("Requete :", placeholder="Ex: remboursement lent, service client")

    col1, col2, col3 = st.columns(3)
    with col1:
        methode = st.radio("Methode", ["BM25", "TF-IDF cosinus"], horizontal=True)
    with col2:
        top_k = st.slider("Nombre de resultats", 3, 20, 5)
    with col3:
        filtre_sent = st.selectbox("Filtrer par sentiment", ["Tous", "positif", "neutre", "negatif"])

    if requete.strip():
        requete_clean = nettoyer_texte(requete)

        # Calcul des scores
        if methode == "BM25":
            scores = bm25.get_scores(requete_clean.split())
        else:
            vq = tfidf_search.transform([requete_clean])
            scores = cosine_similarity(vq, matrice_search).flatten()

        # Tri et filtrage
        candidats = scores.argsort()[-top_k * 5:][::-1]
        candidats = [i for i in candidats if scores[i] > 0]
        if filtre_sent != "Tous":
            candidats = [i for i in candidats if df.iloc[i].get('sentiment', '') == filtre_sent]
        top_indices = candidats[:top_k]

        st.write(f"**{len(top_indices)} resultats pour : \"{requete}\"**")

        for rang, idx in enumerate(top_indices, 1):
            row = df.iloc[idx]
            avis = str(row.get('avis', ''))
            note = int(row.get('note', 3))
            assureur = row.get('assureur', '')
            sentiment = row.get('sentiment', 'neutre')
            score = float(scores[idx])
            couleur = COULEURS.get(sentiment, '#1e2d4a')
            bordure = BORDURES.get(sentiment, '#2d4a7a')
            texte_c = TEXTES.get(sentiment, '#e2e8f0')
            st.markdown(f"""
            <div style="background:{couleur};border-left:4px solid {bordure};
                        padding:14px 16px;border-radius:8px;margin:8px 0;
                        box-shadow:0 2px 8px rgba(0,0,0,0.4);">
                <span style="color:#94a3b8;font-size:.85rem;">
                  <strong style="color:{texte_c};">#{rang} — {assureur}</strong> &nbsp;|&nbsp;
                  Note : {note}/5 &nbsp;|&nbsp; Sentiment : <b style="color:{bordure};">{sentiment}</b> &nbsp;|&nbsp; Score : {score:.3f}
                </span>
                <p style="margin:.5rem 0 0;color:#cbd5e1;line-height:1.5;">{avis[:400]}{'...' if len(avis) > 400 else ''}</p>
            </div>""", unsafe_allow_html=True)


# Page 5 : RAG
elif page == "RAG":
    st.title("RAG — Retrieval Augmented Generation")
    st.write("On recupere les avis les plus pertinents (BM25), puis on genere un resume.")

    requete_rag = st.text_input("Requete :", placeholder="Ex: problemes remboursement sinistre")
    col1, col2 = st.columns(2)
    with col1:
        top_k_rag = st.slider("Documents recuperes", 3, 20, 8)
    with col2:
        nb_phrases_rag = st.slider("Phrases dans la reponse", 2, 5, 3)

    if st.button("Generer une reponse", type="primary"):
        if not requete_rag.strip():
            st.warning("Veuillez entrer une requete.")
        else:
            # Etape 1 : Retrieval (BM25)
            requete_clean = nettoyer_texte(requete_rag)
            scores_rag = bm25.get_scores(requete_clean.split())
            top_idx = scores_rag.argsort()[-top_k_rag:][::-1]
            top_idx = [i for i in top_idx if scores_rag[i] > 0]

            if not top_idx:
                st.warning("Aucun document pertinent trouve.")
            else:
                avis_recuperes = [str(df.iloc[i].get('avis', '')) for i in top_idx]
                notes = [df.iloc[i].get('note', 3) for i in top_idx]
                sentiments = [df.iloc[i].get('sentiment', 'neutre') for i in top_idx]

                # Statistiques sur les documents recuperes
                col1, col2, col3 = st.columns(3)
                col1.metric("Documents trouves", len(top_idx))
                col2.metric("Note moyenne", f"{np.mean(notes):.2f}/5")
                col3.metric("% Negatifs", f"{sum(1 for s in sentiments if s=='negatif')/len(sentiments)*100:.0f}%")

                # Etape 2 : Generation (resume extractif)
                st.markdown("**Reponse generee :**")
                with st.spinner("Generation..."):
                    resume = resumer_avis(avis_recuperes, nb_phrases_rag)
                st.info(resume if resume.strip() else "Resume indisponible.")

                # Sources
                st.write(f"**Sources utilisees (top 5) :**")
                for rang, idx in enumerate(top_idx[:5], 1):
                    row = df.iloc[idx]
                    avis = str(row.get('avis', ''))
                    assureur = row.get('assureur', '')
                    note = int(row.get('note', 3))
                    sentiment = row.get('sentiment', 'neutre')
                    score = float(scores_rag[idx])
                    couleur = COULEURS.get(sentiment, '#1e2d4a')
                    bordure = BORDURES.get(sentiment, '#2d4a7a')
                    texte_c = TEXTES.get(sentiment, '#e2e8f0')
                    st.markdown(f"""
                    <div style="background:{couleur};border-left:4px solid {bordure};
                                padding:14px 16px;border-radius:8px;margin:6px 0;
                                box-shadow:0 2px 8px rgba(0,0,0,0.4);">
                        <span style="color:#94a3b8;font-size:.85rem;">
                          <strong style="color:{texte_c};">#{rang} — {assureur}</strong> &nbsp;|&nbsp;
                          Note : {note}/5 &nbsp;|&nbsp; BM25 : {score:.3f}
                        </span>
                        <p style="margin:.5rem 0 0;color:#cbd5e1;line-height:1.5;">{avis[:300]}{'...' if len(avis) > 300 else ''}</p>
                    </div>""", unsafe_allow_html=True)


# Page 6 : QA
elif page == "QA":
    st.title("Question Answering")
    st.write("Posez une question. Le systeme recupere les avis pertinents et extrait la reponse.")

    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Question :", placeholder="Ex: Quels sont les problemes de remboursement ?")
    with col2:
        assureurs_qa = ["Tous"] + sorted(df['assureur'].dropna().unique().tolist())
        assureur_qa = st.selectbox("Assureur :", assureurs_qa)

    top_k_qa = st.slider("Documents sources", 3, 10, 5)

    if st.button("Repondre", type="primary"):
        if not question.strip():
            st.warning("Veuillez entrer une question.")
        else:
            question_clean = nettoyer_texte(question)

            # Filtrage par assureur si demande
            if assureur_qa != "Tous":
                indices = np.where(df['assureur'] == assureur_qa)[0]
                corpus_filtre = [corpus[i] for i in indices]
                bm25_filtre = BM25Okapi([t.split() for t in corpus_filtre])
                scores_loc = bm25_filtre.get_scores(question_clean.split())
                top_local = scores_loc.argsort()[-top_k_qa:][::-1]
                top_global = [indices[i] for i in top_local if scores_loc[i] > 0]
                scores_qa = {indices[i]: float(scores_loc[i]) for i in range(len(scores_loc))}
            else:
                scores_glob = bm25.get_scores(question_clean.split())
                top_global = scores_glob.argsort()[-top_k_qa:][::-1]
                top_global = [i for i in top_global if scores_glob[i] > 0]
                scores_qa = {int(i): float(scores_glob[i]) for i in top_global}

            if not top_global:
                st.warning("Aucun avis pertinent trouve.")
            else:
                # Extraction de la reponse
                contexte = ' '.join([str(df.iloc[i].get('avis', '')) for i in top_global[:5]])
                reponse = qa_extractif(question, contexte)

                st.markdown("**Reponse :**")
                st.success(reponse)

                # Sources
                st.write(f"**Avis sources ({len(top_global)}) :**")
                for rang, idx in enumerate(top_global[:5], 1):
                    row = df.iloc[idx]
                    avis = str(row.get('avis', ''))
                    assureur = row.get('assureur', '')
                    note = int(row.get('note', 3))
                    sentiment = row.get('sentiment', 'neutre')
                    score = scores_qa.get(int(idx), 0.0)
                    couleur = COULEURS.get(sentiment, '#1e2d4a')
                    bordure = BORDURES.get(sentiment, '#2d4a7a')
                    texte_c = TEXTES.get(sentiment, '#e2e8f0')
                    st.markdown(f"""
                    <div style="background:{couleur};border-left:4px solid {bordure};
                                padding:14px 16px;border-radius:8px;margin:6px 0;
                                box-shadow:0 2px 8px rgba(0,0,0,0.4);">
                        <span style="color:#94a3b8;font-size:.85rem;">
                          <strong style="color:{texte_c};">#{rang} — {assureur}</strong> &nbsp;|&nbsp;
                          Note : {note}/5 &nbsp;|&nbsp; BM25 : {score:.3f}
                        </span>
                        <p style="margin:.5rem 0 0;color:#cbd5e1;line-height:1.5;">{avis[:300]}{'...' if len(avis) > 300 else ''}</p>
                    </div>""", unsafe_allow_html=True)
