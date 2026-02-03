"""
Wakee - App Sourcing
Interface de collecte d'annotations pour le dataset TDAH
"""

import streamlit as st
import requests
import base64
from datetime import datetime
from PIL import Image
import io

# ============================================================================
# CONFIGURATION
# ============================================================================

API_URL = "https://terorra-wakee-api.hf.space"

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Wakee Sourcing",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    /* Titre principal */
    h1 {
        color: #4BE8E0;
        text-align: center;
    }
    
    /* Sous-titres */
    h2, h3 {
        color: #23B1AB;
    }
    
    /* Boutons */
    .stButton>button {
        background-color: #2A7FAF;
        color: white;
        width: 100%;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-size: 1.1em;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #23B1AB;
    }
    
    /* Messages info/success */
    .stSuccess {
        background-color: #015955;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5em;
        color: #4BE8E0;
    }
    
    /* Sliders */
    .stSlider {
        padding: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================

st.markdown("<h1>🧠 Wakee - Annotation Data Collection</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #23B1AB;'>Aidez à améliorer la détection d'émotions pour le TDAH</h3>", unsafe_allow_html=True)
st.markdown("---")

# ============================================================================
# INSTRUCTIONS
# ============================================================================

with st.expander("ℹ️ Comment ça marche ?", expanded=False):
    st.markdown("""
    ### 📋 Instructions
    
    1. **Prenez une photo** avec votre webcam
    2. **Analysez** les scores prédits par l'IA
    3. **Corrigez** les scores avec les sliders si nécessaire
    4. **Validez** pour contribuer à améliorer le modèle
    
    ### 🎯 Les 4 émotions
    
    - **😴 Boredom (Ennui)** : À quel point vous semblez désintéressé
    - **😕 Confusion** : À quel point vous semblez perdu ou confus
    - **🎯 Engagement (Concentration)** : À quel point vous êtes concentré
    - **😤 Frustration** : À quel point vous semblez agacé ou frustré
    
    **Échelle :** 0 = Pas du tout | 3 = Très fortement
    
    ### 🔒 Confidentialité
    
    Vos photos sont stockées de manière anonyme et utilisées uniquement pour améliorer le modèle.
    """)

st.markdown("---")

# ============================================================================
# ÉTAPE 1 : CAPTURE WEBCAM
# ============================================================================

st.markdown("### 📸 Étape 1 : Prenez une photo")

img_file = st.camera_input("Activez votre webcam et prenez une photo")

if img_file is not None:
    
    # ========================================================================
    # ÉTAPE 2 : PRÉDICTION
    # ========================================================================
    
    st.markdown("---")
    st.markdown("### 🤖 Étape 2 : Analyse par l'IA")
    
    with st.spinner("🔄 Analyse en cours..."):
        try:
            # Appel API /predict
            response = requests.post(
                f"{API_URL}/predict",
                files={"file": ("image.jpg", img_file.getvalue(), "image/jpeg")},
                timeout=30
            )
            
            if response.status_code == 200:
                predictions = response.json()
                
                st.success("✅ Analyse terminée !")
                
                # Affichage des prédictions
                st.markdown("#### 📊 Prédictions du modèle :")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "😴 Boredom", 
                        f"{predictions['boredom']:.2f}/3",
                        help="Niveau d'ennui détecté"
                    )
                    st.metric(
                        "😕 Confusion", 
                        f"{predictions['confusion']:.2f}/3",
                        help="Niveau de confusion détecté"
                    )
                
                with col2:
                    st.metric(
                        "🎯 Engagement", 
                        f"{predictions['engagement']:.2f}/3",
                        help="Niveau de concentration détecté"
                    )
                    st.metric(
                        "😤 Frustration", 
                        f"{predictions['frustration']:.2f}/3",
                        help="Niveau de frustration détecté"
                    )
                
                # ============================================================
                # ÉTAPE 3 : CORRECTION UTILISATEUR
                # ============================================================
                
                st.markdown("---")
                st.markdown("### ✏️ Étape 3 : Ajustez les scores si nécessaire")
                
                st.info("💡 **Astuce** : Déplacez les curseurs uniquement si vous pensez que l'IA s'est trompée.")
                
                with st.form("validation_form"):
                    
                    st.markdown("#### 😴 Boredom (Ennui)")
                    user_boredom = st.slider(
                        "À quel point semblez-vous ennuyé ?",
                        min_value=0.0,
                        max_value=3.0,
                        value=float(predictions['boredom']),
                        step=0.5,
                        help="0 = Pas ennuyé du tout | 3 = Très ennuyé",
                        key="boredom"
                    )
                    
                    st.markdown("#### 😕 Confusion")
                    user_confusion = st.slider(
                        "À quel point semblez-vous confus ?",
                        min_value=0.0,
                        max_value=3.0,
                        value=float(predictions['confusion']),
                        step=0.5,
                        help="0 = Pas confus du tout | 3 = Très confus",
                        key="confusion"
                    )
                    
                    st.markdown("#### 🎯 Engagement (Concentration)")
                    user_engagement = st.slider(
                        "À quel point semblez-vous concentré ?",
                        min_value=0.0,
                        max_value=3.0,
                        value=float(predictions['engagement']),
                        step=0.5,
                        help="0 = Pas concentré du tout | 3 = Très concentré",
                        key="engagement"
                    )
                    
                    st.markdown("#### 😤 Frustration")
                    user_frustration = st.slider(
                        "À quel point semblez-vous frustré ?",
                        min_value=0.0,
                        max_value=3.0,
                        value=float(predictions['frustration']),
                        step=0.5,
                        help="0 = Pas frustré du tout | 3 = Très frustré",
                        key="frustration"
                    )
                    
                    st.markdown("---")
                    
                    # Bouton de validation
                    submitted = st.form_submit_button(
                        "✅ Valider et envoyer l'annotation",
                        type="primary",
                        use_container_width=True
                    )
                    
                    if submitted:
    
                        # ================================================
                        # ÉTAPE 4 : ENVOI À L'API
                        # ================================================
                        
                        with st.spinner("📤 Envoi en cours..."):
                            try:
                                # ✅ CHANGEMENT : Plus de base64, envoi direct du fichier
                                files = {
                                    'file': ('image.jpg', img_file.getvalue(), 'image/jpeg')
                                }
                                
                                # ✅ CHANGEMENT : Les données dans 'data' au lieu de 'json'
                                data = {
                                    'predicted_boredom': predictions['boredom'],
                                    'predicted_confusion': predictions['confusion'],
                                    'predicted_engagement': predictions['engagement'],
                                    'predicted_frustration': predictions['frustration'],
                                    'user_boredom': user_boredom,
                                    'user_confusion': user_confusion,
                                    'user_engagement': user_engagement,
                                    'user_frustration': user_frustration
                                }
                                
                                # ✅ CHANGEMENT : files= et data= au lieu de json=
                                insert_response = requests.post(
                                    f"{API_URL}/insert",
                                    files=files,
                                    data=data,
                                    timeout=60
                                )
                                
                                # Le reste est IDENTIQUE à ton code
                                if insert_response.status_code == 200:
                                    result = insert_response.json()
                                    
                                    # Succès !
                                    st.balloons()
                                    st.success(f"🎉 **{result['message']}**")
                                    st.info(f"📋 Image ID : `{result['img_name']}`")
                                    
                                    st.markdown("---")
                                    st.markdown("""
                                    ### 🙏 Merci pour votre contribution !
                                    
                                    Votre annotation va aider à :
                                    - ✅ Améliorer la précision du modèle
                                    - ✅ Diversifier le dataset
                                    - ✅ Mieux accompagner les personnes TDAH
                                    
                                    **Vous pouvez maintenant prendre une nouvelle photo ou fermer cette page.**
                                    """)
                                    
                                else:
                                    st.error(f"❌ Erreur lors de l'envoi : {insert_response.status_code}")
                                    st.error(f"Détails : {insert_response.text}")
                            
                            except requests.exceptions.Timeout:
                                st.error("⏱️ Timeout : L'envoi a pris trop de temps. Veuillez réessayer.")
                            
                            except Exception as e:
                                st.error(f"❌ Erreur inattendue : {str(e)}")
            
            else:
                st.error(f"❌ Erreur lors de l'analyse : {response.status_code}")
                st.error(f"Détails : {response.text}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Impossible de se connecter à l'API. Vérifiez que l'API est en ligne.")
        
        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout : L'analyse a pris trop de temps. Veuillez réessayer.")
        
        except Exception as e:
            st.error(f"❌ Erreur inattendue : {str(e)}")

else:
    st.info("👆 Cliquez sur la caméra ci-dessus pour prendre une photo")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #CCCCCC; margin-top: 2rem;">
    <p>Développé avec 💙 par <strong>Terorra</strong></p>
    <p>Certification AIA Lead MLOps</p>
    <p style="font-size: 0.8em;">
        <a href="https://huggingface.co/spaces/Terorra/wakee-api" target="_blank" style="color: #4BE8E0;">API</a> • 
        <a href="https://github.com/Terorra/wakee-reloaded" target="_blank" style="color: #4BE8E0;">GitHub</a> • 
        <a href="https://huggingface.co/Terorra/wakee-reloaded" target="_blank" style="color: #4BE8E0;">Modèle</a>
    </p>
</div>
""", unsafe_allow_html=True)
