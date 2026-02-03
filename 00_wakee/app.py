import streamlit as st
import cv2
from PIL import Image
import time
from collections import deque, Counter
import numpy as np
from datetime import datetime, timedelta
import warnings
from dotenv import load_dotenv

import cnn
import llm

warnings.filterwarnings("ignore")
load_dotenv()

# ============================================================================
# CONFIGURATION POPUP SOURCING
# ============================================================================

SOURCING_URL = "https://terorra-wakee-sourcing.hf.space"
POPUP_INTERVAL_HOURS = 4  # Popup toutes les 4 heures
POPUP_AUTO_DISMISS_MINUTES = 10  # Auto-dismiss après 10 min d'inactivité

# ============================================================================
# CSS CUSTOM
# ============================================================================

custom_css = """
<style>
/* Changer la couleur des titres H1 */
h1 {
    color: #4BE8E0;
    text-align: center;
}

/* Changer la couleur des titres H2 */
h2 {
    color: #23B1AB;
}

/* Cibler les boutons Streamlit */
.stButton>button {
    background-color: #2A7FAF;
    color: white;
}

/* Cibler les liens */
a {
    color: #9DD4F3;
}

/* Style pour l'encart central de Wakee */
.wakee-message-container {
    background-color: #015955;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
    text-align: center;
    margin-top: 20px;
}
.wakee-message-container h3 {
    color: #97FBF6;
    font-size: 1.8em;
    margin-bottom: 15px;
}
.llm-output-text {
    color: #FFFFFF;
    font-size: 1.6em;
    font-weight: bold;
    line-height: 1.4;
}

/* Popup contribution styling */
.contribution-popup {
    background: linear-gradient(135deg, #015955 0%, #023844 100%);
    padding: 25px;
    border-radius: 15px;
    border-left: 5px solid #4BE8E0;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    margin: 20px 0;
}
.contribution-popup h3 {
    color: #4BE8E0;
    margin: 0 0 15px 0;
}
.contribution-popup p {
    color: #FFFFFF;
    margin: 10px 0;
    line-height: 1.6;
}
.contribution-popup strong {
    color: #97FBF6;
}
.contribution-popup small {
    color: #97FBF6;
    font-size: 0.85em;
}

.stColumn {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

# Popup sourcing
if 'last_popup_time' not in st.session_state:
    st.session_state.last_popup_time = None

if 'popup_dismissed' not in st.session_state:
    st.session_state.popup_dismissed = False

if 'popup_shown_time' not in st.session_state:
    st.session_state.popup_shown_time = None

# Wakee session
if "start_time" not in st.session_state:
    st.session_state.start_time = None

# ============================================================================
# FONCTIONS POPUP
# ============================================================================

def should_show_popup():
    """
    Vérifie si le popup doit être affiché
    Conditions :
    - Jamais affiché OU
    - Dernier affichage il y a plus de 4h
    - Pas déjà affiché dans cette session
    """
    # Si déjà dismissed dans cette session
    if st.session_state.popup_dismissed:
        return False
    
    # Premier affichage
    if st.session_state.last_popup_time is None:
        return True
    
    # Vérifier intervalle de 4h
    time_since_last = datetime.now() - st.session_state.last_popup_time
    return time_since_last > timedelta(hours=POPUP_INTERVAL_HOURS)

def should_auto_dismiss_popup():
    """
    Vérifie si le popup doit être auto-dismissed (10 min d'inactivité)
    """
    if st.session_state.popup_shown_time is None:
        return False
    
    time_since_shown = datetime.now() - st.session_state.popup_shown_time
    return time_since_shown > timedelta(minutes=POPUP_AUTO_DISMISS_MINUTES)

def show_sourcing_popup():
    """
    Affiche le popup d'invitation à contribuer
    Auto-dismiss après 10 min d'inactivité
    """
    # Initialise le timer si pas déjà fait
    if st.session_state.popup_shown_time is None:
        st.session_state.popup_shown_time = datetime.now()
    
    # Vérifie auto-dismiss
    if should_auto_dismiss_popup():
        st.session_state.popup_dismissed = True
        st.session_state.last_popup_time = datetime.now()
        st.session_state.popup_shown_time = None
        st.rerun()
        return
    
    # Calcul temps restant avant auto-dismiss
    time_since_shown = datetime.now() - st.session_state.popup_shown_time
    remaining_minutes = POPUP_AUTO_DISMISS_MINUTES - int(time_since_shown.total_seconds() / 60)
    
    st.markdown(f"""
    <div class="contribution-popup">
        <h3>🙏 Aidez à améliorer Wakee !</h3>
        <p>Vous utilisez Wakee depuis un moment. Voulez-vous contribuer à améliorer 
        la précision du modèle en validant quelques détections ?</p>
        <p><strong>✨ Ça prend 2 minutes et ça aide énormément !</strong></p>
        <p style="font-size: 0.9em; color: #97FBF6; margin-top: 15px;">
            💡 Vos annotations permettent d'améliorer la détection pour tous
        </p>
        <small>⏱️ Ce message disparaîtra automatiquement dans {remaining_minutes} min</small>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("✅ Contribuer", use_container_width=True, type="primary", key="popup_contribute"):
            st.session_state.last_popup_time = datetime.now()
            st.session_state.popup_dismissed = True
            st.session_state.popup_shown_time = None
            st.markdown(f"""
            <script>
                window.open('{SOURCING_URL}', '_blank');
            </script>
            """, unsafe_allow_html=True)
            st.success("🎉 Merci ! L'app s'ouvre dans un nouvel onglet.")
            time.sleep(2)
            st.rerun()
    
    with col2:
        if st.button("⏭️ Rappeler dans 4h", use_container_width=True, key="popup_later"):
            st.session_state.last_popup_time = datetime.now()
            st.session_state.popup_dismissed = True
            st.session_state.popup_shown_time = None
            st.rerun()

# ============================================================================
# FONCTIONS WAKEE (ORIGINALES)
# ============================================================================

deque_length = 100
history = deque([], maxlen=deque_length)
last_action_time = time.time()
FRAME_SKIP_SECONDS = 0.1
prev_frame_time = 0
frame_count = 0

def showfps(prev_time):
    """Performance measurement"""
    new_time = time.time()
    fps = 1 / (new_time - prev_time + 1e-8)
    return new_time, int(fps)

def get_response_from_cnn(frame):
    """CNN emotion detection"""
    pilimage = Image.fromarray(frame).convert("RGB")
    cnn_predict = cnn.get_emotion(pilimage)[0].tolist()
    dict_cnn = {"boredom": cnn_predict[0], "confusion": cnn_predict[1],
                "engagement": cnn_predict[2], "frustration": cnn_predict[3]}
    cnn_engagement = dict_cnn["engagement"]
    cnn_boredom = dict_cnn["boredom"]
    cnn_confusion = dict_cnn["confusion"]
    cnn_frustration = dict_cnn["frustration"]

    if cnn_engagement < 1:
        return "disengagement"
    elif cnn_frustration > 0.5:
        return "frustration"
    elif cnn_confusion > 0.61:
        return "confusion"
    elif cnn_boredom > 1.05:
        return "boredom"
    else:
        return "incertitude"

def evaluate_response(history):
    """Extract most common emotion from history"""
    if not history:
        return "incertitude"
    return Counter(history).most_common(1)[0][0]

def single_use_slider(key_prefix: str = "default_slider", title: str = "Sélectionnez une valeur", options=None):
    """Slider for work time selection"""
    if options is None:
        options = [f"{i:02d}:00" for i in range(24)]

    confirmed_key = f"{key_prefix}_confirmed_value"
    disabled_key = f"{key_prefix}_disabled"

    if confirmed_key not in st.session_state:
        st.session_state[confirmed_key] = None
    if disabled_key not in st.session_state:
        st.session_state[disabled_key] = False

    if not st.session_state[disabled_key]:
        st.write(f"Je choisis mon temps de travail et je confirme.")
        selected_value = st.select_slider(
            title,
            options=options,
            key=f"{key_prefix}_slider_widget"
        )

        if st.button(f'Confirmer mon temps de travail', key=f"{key_prefix}_confirm_button"):
            st.session_state[confirmed_key] = selected_value
            st.session_state[disabled_key] = True
            st.rerun()
    else:
        st.success(f"Je vais travailler pendant **{st.session_state[confirmed_key]}** min. C'est parti !")

    return st.session_state[confirmed_key]

def render_wakee_message_container(message: str):
    """HTML generator for suggestions"""
    return f"""
    <div class="wakee-message-container">
        <h3>💬 Suggestions de Wakee :</h3>
        <p class="llm-output-text">{message}</p>
    </div>
    """

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(page_title="WAKEE", layout="wide")

st.markdown("<h1 style='text-align: center; color: #4BE8E0;'>Je travaille avec WAKEE !</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #23B1AB;'>Reconnaissance des émotions & Recommendation pour le TDAH 🧠</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #FFFFFF;'>🤖 W.A.K.E.E. : Work Assistant with Kindness & Emotional Empathy 🤗</p>", unsafe_allow_html=True)

# Info modèle HF
with st.expander("ℹ️ À propos du modèle", expanded=False):
    st.markdown("""
    ### 🤖 Modèle d'IA
    
    - **Source** : [HuggingFace Model Hub](https://huggingface.co/Terorra/wakee-reloaded)
    - **Architecture** : EfficientNet B4 (ONNX)
    - **Dataset** : DAiSEE (réentraîné avec vos contributions)
    - **Mise à jour** : Automatique depuis HF Hub
    
    Le modèle est téléchargé automatiquement au premier lancement et mis en cache localement.
    """)

st.markdown("---")

# ============================================================================
# POPUP CONTRIBUTION (toutes les 4h, auto-dismiss 10min)
# ============================================================================

if should_show_popup():
    show_sourcing_popup()
    st.markdown("---")

# ============================================================================
# INTERFACE PRINCIPALE (CODE ORIGINAL)
# ============================================================================

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.markdown("<h3 style='text-align: center; color: #23B1AB;'>⏱️ Choix du temps de travail</h3>", unsafe_allow_html=True)

    selected_time_confirmed = single_use_slider(
        key_prefix="time_selection",
        title="Combien de temps je veux travailler (en minutes) : ",
        options=[f"{i:02d}" for i in range(15, 135, 15)]
    )

    if selected_time_confirmed:
        st.info("Je peux allumer ma caméra et démarrer ma session 🦾​")

    if st.button('Réinitialiser & changer mon temps de travail'):
        st.session_state["time_selection_confirmed_value"] = None
        st.session_state["time_selection_disabled"] = False
        st.session_state.start_time = None
        st.rerun()

    st.markdown("<h3 style='text-align: center; color: #23B1AB;'>⌛ Temps écoulé</h3>", unsafe_allow_html=True)
    progress_bar_placeholder = st.empty()
    progress_text_placeholder = st.empty()

    if st.button('Arrêter l\'application', type="primary"):
        st.warning("Application arrêtée. Pour redémarrer, relancez le script.")
        st.stop()

with col2:
    st.markdown("<h3 style='text-align: center; color: #23B1AB;'>🎥 Ma camera</h3>", unsafe_allow_html=True)

    start_button = st.toggle("Activer ma camera 🔴​")

    image_display = st.empty()
    stats_display = st.empty()
    emotion_display = st.empty()

with col3:
    llm_container_placeholder = st.empty()
    st.image("WAKEE_image.png", use_container_width=True)

# ============================================================================
# CORE SCRIPT (CODE ORIGINAL)
# ============================================================================

if start_button:
    if st.session_state.start_time is None:
        st.session_state.start_time = time.time()
        st.session_state.time_remaining_message_sent = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("⚠️ Impossible d'accéder à la caméra.")
    else:
        base_message = "Prêt à t'aider ! J'analyse tes émotions et je t'aiderai au besoin 😊​"
        current_llm_message = base_message
        llm_container_placeholder.markdown(render_wakee_message_container(f'💬 {current_llm_message}'), unsafe_allow_html=True)

        total_work_time_minutes = int(selected_time_confirmed) if selected_time_confirmed else 0
        total_work_time_seconds = total_work_time_minutes * 60

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⌚ Erreur de lecture de la caméra.")
                break

            frame_count += 1
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            prev_frame_time, fps = showfps(prev_frame_time)
            emotion = get_response_from_cnn(rgb_frame)
            history.append(emotion)

            elapsed_time_seconds = time.time() - st.session_state.start_time
            
            if total_work_time_seconds > 0:
                progress_percentage = min(elapsed_time_seconds / total_work_time_seconds, 1.0)
            else:
                progress_percentage = 0.0

            progress_bar_placeholder.progress(progress_percentage)
            
            remaining_seconds = max(0, total_work_time_seconds - elapsed_time_seconds)
            mins = int(remaining_seconds // 60)
            secs = int(remaining_seconds % 60)
            
            progress_text_placeholder.text(f"Temps restant : {mins:02d}m {secs:02d}s")

            if elapsed_time_seconds >= total_work_time_seconds and total_work_time_seconds > 0 and not st.session_state.time_remaining_message_sent:
                llm_container_placeholder.markdown(render_wakee_message_container("Félicitations ! Votre temps de travail est terminé. Prenez une pause bien méritée ! 🎉"), unsafe_allow_html=True)
                st.session_state.time_remaining_message_sent = True

            with col2:
                image_display.image(rgb_frame, channels="RGB")
                stats_display.markdown(f"**🧮 Frame :** {frame_count}  |  **⚡ FPS :** {fps}")
                emotion_display.markdown(f"**🧠 Dernière émotion détectée :** `{emotion}`")

            if len(history) == deque_length:
                action = evaluate_response(history)
                history.clear()

                if time.time() - last_action_time >= 10 and action != "incertitude":
                    last_action_time = time.time()
                    message = llm.get_recommendation(action)
                    current_llm_message = message
                elif action == "incertitude":
                    current_llm_message = base_message
            
            if not st.session_state.get("time_remaining_message_sent", False):
                llm_container_placeholder.markdown(render_wakee_message_container(f'{current_llm_message}'), unsafe_allow_html=True)

            time.sleep(FRAME_SKIP_SECONDS)

        cap.release()
else:
    with col2:
        image_display.empty()
        stats_display.empty()
        emotion_display.empty()
    
    with col1:
        progress_bar_placeholder.progress(0)
        progress_text_placeholder.text("Temps restant : 00m 00s")
    
    st.session_state.start_time = None
    st.session_state.time_remaining_message_sent = False

    with col3:
        llm_container_placeholder.markdown(render_wakee_message_container("Démarre ta session en activant la caméra 👌​"), unsafe_allow_html=True)

# ============================================================================
# CREDITS
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px; color: #CCCCCC;">
        <p>Développé avec 💙 par :</p>
        <p><strong>Albert ROMANO, Asma RHALMI, Jeremy MARIAGE, Manon FAEDY</strong></p>
        <p style="font-size: 0.9em; margin-top: 10px;">
            Modèle hébergé sur <a href="https://huggingface.co/Terorra/wakee-reloaded" target="_blank" style="color: #4BE8E0;">HuggingFace Model Hub</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
