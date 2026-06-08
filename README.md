# TikTok Foot Generator

Outil local : tu colles un lien YouTube (résumé de match), tu choisis l'extrait et le style, et l'app te renvoie une vidéo verticale **1080×1920** prête pour TikTok.

## Prérequis

- **Python 3.10+** — https://www.python.org/downloads/
- **ffmpeg** dans le PATH — https://ffmpeg.org/download.html (ou `winget install Gyan.FFmpeg`)

## Installation

```powershell
cd r-sum-vid-o1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Lancement

```powershell
python app.py
```

Puis ouvre **http://127.0.0.1:5000** dans ton navigateur.

## Utilisation

1. Colle le lien YouTube du résumé
2. Choisis le **début** (en secondes) et la **durée** (laisse vide pour tout garder)
3. Choisis un style :
   - **Fond flouté** — l'action centrée sur un fond flou (le plus TikTok)
   - **Recadré** — zoom centré, perd les côtés
   - **Bandes noires** — vidéo complète avec bandes noires
4. Clique sur **Générer**
5. Aperçu, puis **Télécharger**

## Conseils virage TikTok

- Vise **15–60 secondes**, un seul temps fort
- Démarre **à l'action** : pas de bandeau de chaîne, pas de logo
- Le format **Fond flouté** est celui qui marche le mieux sur des résumés 16:9

## ⚖️ Note importante

Tu es responsable des droits sur les vidéos que tu reposterais. Les droits FIFA/UEFA/diffuseurs sont strictement protégés et TikTok détecte/démonétise/supprime régulièrement les contenus repostés. Utilise des extraits courts, des sources libres de droits ou tes propres images quand c'est possible.
