---
title: Wakee Sourcing
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
license: mit
---

# 🧠 Wakee - Annotation Data Collection

Interface de collecte d'annotations pour améliorer la détection d'émotions TDAH.

## 🎯 Objectif

Cette application permet de collecter des annotations humaines pour enrichir le dataset du modèle Wakee et améliorer sa précision sur différents morphotypes.

## 🚀 Comment utiliser

1. **Prenez une photo** avec votre webcam
2. **Analysez** les 4 scores prédits par l'IA :
   - 😴 Boredom (Ennui)
   - 😕 Confusion
   - 🎯 Engagement (Concentration)
   - 😤 Frustration
3. **Corrigez** les scores si nécessaire avec les sliders
4. **Validez** pour contribuer à améliorer le modèle

## 🔄 Workflow

```
Utilisateur → Photo webcam → API /predict → Affichage scores
                                    ↓
              Correction sliders → API /insert → R2 + NeonDB
```

## 🏗️ Architecture

- **Frontend** : Streamlit
- **API Backend** : [Terorra/wakee-api](https://huggingface.co/spaces/Terorra/wakee-api)
- **Stockage images** : Cloudflare R2
- **Base de données** : NeonDB (PostgreSQL)
- **Modèle** : [Terorra/wakee-reloaded](https://huggingface.co/Terorra/wakee-reloaded)

## 📊 Les 4 émotions

### 😴 Boredom (Ennui)
Niveau de désintérêt ou d'ennui visible sur le visage.

### 😕 Confusion
Niveau de confusion ou d'incompréhension visible.

### 🎯 Engagement (Concentration)
Niveau de concentration ou d'engagement dans la tâche.

### 😤 Frustration
Niveau de frustration ou d'agacement visible.

**Échelle** : 0 (pas du tout) → 3 (très fortement)

## 🔒 Confidentialité

- Les photos sont stockées de manière anonyme
- Utilisées uniquement pour améliorer le modèle
- Pas de données personnelles collectées
- Conformité RGPD

## 🔗 Liens

- [Code source GitHub](https://github.com/Terorra/wakee-reloaded)
- [API Endpoint](https://huggingface.co/spaces/Terorra/wakee-api)
- [Modèle HuggingFace](https://huggingface.co/Terorra/wakee-reloaded)

## 👨‍💻 Auteur

**Terorra** - Certification AIA Lead MLOps

---

**Développé avec 💙 pour aider les personnes atteintes de TDAH**
