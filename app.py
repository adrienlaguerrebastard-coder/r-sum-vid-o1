import array
import os
import re
import sys
import shutil
import tempfile
import subprocess
import threading
import unicodedata
import uuid
from pathlib import Path

import yt_dlp
from flask import Flask, jsonify, render_template, request, send_file

BASE_DIR = Path(__file__).parent
DOWNLOADS_DIR = BASE_DIR / "downloads"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOWNLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

AUDIO_SR = 1000
PRE_PEAK_SEC = 6
POST_PEAK_SEC = 9
MIN_CLIP_SEC = 5
MAX_CLIPS = 24
GROUP_GAP_SEC = 3
PEAK_PERCENTILE = 0.68   # plus bas = plus de pics audio détectés (plus d'occasions)

WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")  # base<small<medium (+ précis, + lent)
GOAL_PRE_SEC = 18   # on remonte large depuis la transition de score (l'action est 3-20s avant,
GOAL_POST_SEC = 3   # variable) ; le clip FINIT après la transition → le but est toujours dedans
GOAL_MIN_SEC = 6         # longueur mini d'un clip de but quand on doit en caser beaucoup
ACTION_PRE_SEC = 4
ACTION_POST_SEC = 6
GOAL_DEDUPE_SEC = 14    # < : sinon 2 buts rapprochés (match à bcp de buts) sont fusionnés → buts perdus
GOALS_FILL_FRAC = 0.72  # si les buts remplissent déjà cette fraction de la cible → pas d'occasions (priorité buts)
ACTION_EXCLUDE_SEC = 12  # fenêtre après un but où les "actions" sont ignorées (= replays du but)
TARGET_MAX_SEC = 300

# Détection des ralentis (replays slow-motion) par analyse du mouvement vidéo.
SLOWMO_FPS = 8           # échantillonnage pour la passe de mouvement
SLOWMO_WIDTH = 240       # downscale pour aller vite
SLOWMO_RATIO = 0.65      # seuil = ratio * mouvement MÉDIAN (capte replays/gros plans des reels)
SLOWMO_MIN_SEC = 3       # une zone < cette durée n'est pas considérée comme un ralenti
SLOWMO_GAP_SEC = 1       # tolérance de trou pour fusionner deux secondes basses
SLOWMO_KEEP_SEC = 3      # durée max du ralenti conservé (1 replay court par but)
SLOWMO_MAX_FRAC = 0.40   # si > 40% de la vidéo est flaggée, on désactive (détecteur non fiable)

# --- Taxonomie des événements + priorités (spec) ---------------------------
# Priorité : plus le chiffre est BAS, plus l'événement est important (gardé en 1er).
PRIORITY = {
    "goal": 1,
    "penalty": 2,
    "redcard": 3,
    "chance": 4,    # occasion franche
    "save": 5,      # arrêt décisif
    "audio": 5,     # pic de clameur sans mot-clé (intensité / occasion probable)
    "celebration": 6,
    "replay": 5,    # ralenti court d'un but (1 seul/but, ≤3s) → accompagne le but
}
MANDATORY_TYPES = {"goal", "penalty", "redcard"}  # toujours présents, quelle que soit la durée

GOAL_KEYWORDS = [
    "but ", "buts ", "but,", "but!", "but.", "but :", "but de", "but pour",
    "marque", "marqué", "marquer",
    "ouvre le score", "double la mise", "triple la mise", "creuse l'écart",
    "doublé", "triplé", "le double", "ont été punis", "ont ete punis", "punis",
    "égalise", "égalisation", "égalisateur", "réduit le score", "réduit l'écart",
    "réduit la marque", "revient au score", "revient dans le match", "recolle",
    "renverse", "retournement", "contre son camp", "csc", "trompe son gardien",
    "trouve la faille", "fait mouche", "inscrit", "convertit", "conclut",
    "filets", "filet", "lucarne", "dans les buts", "au fond", "dans la lucarne",
    "goal", "goooal", "gooal", "goalll", "scores", "scored",
    "transformé", "transforme le penalty",
]
PENALTY_KEYWORDS = [
    "penalty", "pénalty", "péno", "peno", "coup de pied de réparation",
    "point de penalty", "faute dans la surface",
]
REDCARD_KEYWORDS = [
    "carton rouge", "rouge direct", "expulsion", "expulsé", "expulse",
    "deuxième jaune", "second jaune", "réduit à dix", "infériorité numérique",
]
SAVE_KEYWORDS = [
    "arrêt", "parade", "sauvetage", "détourne", "repousse", "claque",
    "s'interpose", "sort le ballon", "main ferme", "réflexe", "envoie en corner",
]
CHANCE_KEYWORDS = [
    "frappe", "tir", "reprise", "volée", "demi-volée", "tête", "occasion",
    "coup franc", "corner décisif", "poteau", "barre", "montant", "passe à côté",
    "manque le cadre", "tout proche", "à côté", "face au but", "seul devant",
]
EXCITEMENT_KEYWORDS = [
    "incroyable", "magnifique", "magistral", "splendide", "merveilleux",
    "fantastique", "extraordinaire", "spectaculaire", "superbe",
    "remarquable", "phénoménal", "exceptionnel", "génial", "sublime",
    "quel but", "quel arrêt", "quelle frappe", "quelle action", "quelle reprise",
    "what a goal", "amazing", "unbelievable", "incredible",
]

GOAL_SCORE = 10
EXCITEMENT_SCORE = 3
ACTION_SCORE = 1
MIN_SEGMENT_SCORE = 1

# --- Détection des buts par OCR du score (source de vérité, priorité absolue) ---
# On OCR le chiffre DOMICILE et le chiffre EXTÉRIEUR SÉPARÉMENT (on exclut le point
# central, lu sinon comme un chiffre). PROFIL PAR CHAÎNE : chaque diffuseur place son
# bandeau différemment. Positions = fractions de l'image. `region_*` = zone du bandeau
# pour l'option "afficher le score en haut au centre". À calibrer par chaîne.
SCOREBOARDS = {
    "tf1": {  # TF1 : bandeau haut-gauche, chiffres ~0.205 / 0.237 ; score maj ~2s après l'action
        "home_x": 0.205, "away_x": 0.237, "digit_y": 0.065, "digit_h": 0.05, "digit_w": 0.022,
        "region_x": 0.06, "region_y": 0.04, "region_w": 0.28, "region_h": 0.09, "score_delay": 2,
    },
    "lequipe": {  # L'Équipe : chiffres ~0.142 / 0.169 (plus bas) ; score maj ~13s après l'action
        "home_x": 0.142, "away_x": 0.169, "digit_y": 0.086, "digit_h": 0.058, "digit_w": 0.024,
        "region_x": 0.0, "region_y": 0.03, "region_w": 0.40, "region_h": 0.10, "score_delay": 13,
    },
}
SCORE_OCR_STEP = 1.0       # secondes entre deux lectures du score
SCORE_STABLE = 2           # un score doit apparaître ≥ ce nb de lectures pour être validé
# Signal 3 : "roar" du public (montée sonore soutenue après un but).
ROAR_WIN_SEC = 12
ROAR_FACTOR = 1.6          # un instant est "fort" s'il dépasse 1.6× la médiane
ROAR_FRACTION = 0.55       # ... et il faut que la majorité de la fenêtre le soit
# Signal 3 : "roar" du public (montée sonore soutenue après un but).
ROAR_WIN_SEC = 12
ROAR_FACTOR = 1.6          # un instant est "fort" s'il dépasse 1.6× la médiane
ROAR_FRACTION = 0.55       # ... et il faut que la majorité de la fenêtre le soit
# Poids de confiance par signal + seuil de confirmation d'un but.
W_OCR = 100
W_TXT = 32
W_ROAR = 26
GOAL_CONFIRM = 45          # but confirmé si confiance >= ce seuil (ou OCR présent)
MAX_PLAUSIBLE_GOALS = 12   # garde-fou : au-delà, l'OCR du score déraille → ignoré


def _strip_accents(text: str) -> str:
    return unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii").lower()


# --- NBA / basket : temps forts repérés au commentaire (EN + FR) -----------
NBA_BIG_KEYWORDS = [  # gros moments
    "dunk", "slam", "poster", "alley-oop", "alley oop", "windmill", "throw it down",
    "throws it down", "flush", "jam ", "and-one", "and one", "buzzer", "game winner",
    "game-winner", "dagger", "clutch", "four-point", "four point play", "oop",
    "claquette", "dunke", "alleyoop", "panier de la gagne", "au buzzer",
]
NBA_THREE_KEYWORDS = [
    "three", "3-pointer", "three-pointer", "triple", "from downtown", "for three",
    "beyond the arc", "from deep", "trois points", "a trois points", "derriere l'arc",
    "ficelle", "primeur",
]
NBA_DEF_KEYWORDS = [
    "block", "blocked", "swat", "rejection", "steal", "stuffed", "denied",
    "contre", "interception", "chien", "vol de balle",
]
NBA_EXC_KEYWORDS = [
    "incredible", "unbelievable", "oh my", "are you kidding", "what a", "wow",
    "amazing", "ridiculous", "insane", "huge", "are you serious", "no way",
    "incroyable", "enorme", "quelle action", "magnifique", "monstrueux", "enormissime",
]

GOAL_KW_NORM = [_strip_accents(k) for k in GOAL_KEYWORDS]
NBA_BIG_NORM = [_strip_accents(k) for k in NBA_BIG_KEYWORDS]
NBA_THREE_NORM = [_strip_accents(k) for k in NBA_THREE_KEYWORDS]
NBA_DEF_NORM = [_strip_accents(k) for k in NBA_DEF_KEYWORDS]
NBA_EXC_NORM = [_strip_accents(k) for k in NBA_EXC_KEYWORDS]
PENALTY_KW_NORM = [_strip_accents(k) for k in PENALTY_KEYWORDS]
REDCARD_KW_NORM = [_strip_accents(k) for k in REDCARD_KEYWORDS]
SAVE_KW_NORM = [_strip_accents(k) for k in SAVE_KEYWORDS]
CHANCE_KW_NORM = [_strip_accents(k) for k in CHANCE_KEYWORDS]
EXCITEMENT_KW_NORM = [_strip_accents(k) for k in EXCITEMENT_KEYWORDS]

_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper_model():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(
                WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
            )
        return _whisper_model


def _has_kw(text_norm: str, kws: list[str]) -> bool:
    return any(kw in text_norm for kw in kws)


def classify_segment(text: str) -> tuple[str | None, float]:
    """Classe un segment de commentaire en (type d'événement, score d'intensité).

    Le type retenu est le PLUS prioritaire présent dans le texte. Le score sert à
    départager les candidats d'une même priorité (boosté par les mots d'excitation)."""
    tn = " " + _strip_accents(text) + " "
    excite = sum(EXCITEMENT_SCORE for kw in EXCITEMENT_KW_NORM if kw in tn)

    if _has_kw(tn, GOAL_KW_NORM):
        return "goal", GOAL_SCORE + excite
    if _has_kw(tn, REDCARD_KW_NORM):
        return "redcard", GOAL_SCORE + excite
    if _has_kw(tn, PENALTY_KW_NORM):
        return "penalty", GOAL_SCORE + excite
    if _has_kw(tn, SAVE_KW_NORM):
        return "save", ACTION_SCORE + excite + 2
    if _has_kw(tn, CHANCE_KW_NORM):
        return "chance", ACTION_SCORE + excite + 1
    if excite:
        return "chance", excite  # excitation seule = occasion probable
    return None, 0.0


def classify_segment_nba(text: str) -> tuple[str | None, float]:
    """Classifieur BASKET : repère les temps forts (dunk, 3pts, contre, clutch…).

    Pas de notion de "but" (on marque sans arrêt) : tout est une "occasion" notée par
    intensité, et la sélection garde les meilleurs moments répartis sur le match."""
    tn = " " + _strip_accents(text) + " "
    exc = sum(EXCITEMENT_SCORE for kw in NBA_EXC_NORM if kw in tn)
    if _has_kw(tn, NBA_BIG_NORM):
        return "chance", 12 + exc        # dunk / clutch / buzzer / and-one
    if _has_kw(tn, NBA_THREE_NORM):
        return "chance", 9 + exc          # 3 points
    if _has_kw(tn, NBA_DEF_NORM):
        return "save", 7 + exc            # contre / interception
    if exc:
        return "chance", 4 + exc          # gros pic d'enthousiasme seul
    return None, 0.0


def transcribe_video(path: Path, job_id: str, video_duration: float,
                     language: str | None = None) -> list[dict]:
    model = get_whisper_model()
    # priorité : env WHISPER_LANG > langue du sport (param) > auto-détection
    lang = os.environ.get("WHISPER_LANG") or language or None
    segments_iter, info = model.transcribe(
        str(path),
        language=lang,
        beam_size=5,             # recherche plus large = transcription plus fidèle
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400),
        condition_on_previous_text=False,
    )
    segments: list[dict] = []
    ref_duration = video_duration or info.duration or 1
    for seg in segments_iter:
        segments.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text})
        pct = 50 + int(seg.end / ref_duration * 30)
        update_job(job_id, progress=min(pct, 80))
    return segments


def compute_motion_per_sec(path: Path, duration: float) -> list[float] | None:
    """Énergie de mouvement moyenne par seconde (diff entre images consécutives).

    Une passe ffmpeg basse résolution : tblend en mode différence puis signalstats
    donne, par image, la luminance moyenne de la différence ≈ quantité de mouvement.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-an",
        "-vf",
        f"fps={SLOWMO_FPS},scale={SLOWMO_WIDTH}:-2,tblend=all_mode=difference,"
        f"signalstats,metadata=print:file=-",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not result.stdout:
        return None

    secs = max(1, int(duration) + 1) if duration > 0 else 1
    sums = [0.0] * secs
    counts = [0] * secs
    cur_t = 0.0
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("frame:"):
            m = re.search(r"pts_time:([\d.]+)", line)
            if m:
                cur_t = float(m.group(1))
        elif line.startswith("lavfi.signalstats.YAVG="):
            try:
                val = float(line.split("=", 1)[1])
            except ValueError:
                continue
            idx = min(secs - 1, int(cur_t))
            sums[idx] += val
            counts[idx] += 1
    if not any(counts):
        return None
    return [sums[i] / counts[i] if counts[i] else 0.0 for i in range(secs)]


def detect_slowmo_regions(motion: list[float]) -> list[tuple[float, float]]:
    """Zones (en secondes) de mouvement nettement plus faible que le jeu en direct.

    Volontairement CONSERVATEUR : seuil bas (relatif à la médiane) pour ne viser que
    les vrais creux (replays/plans figés), et AUTO-DÉSACTIVÉ si plus de SLOWMO_MAX_FRAC
    de la vidéo est flaggée — signe que la mesure de mouvement ne discrimine pas bien
    sur cette source (mieux vaut ne rien couper que tout déchiqueter).
    """
    vals = [m for m in motion if m > 0]
    if len(vals) < 6:
        return []
    s = sorted(vals)
    median = s[len(s) // 2]
    if median <= 0:
        return []
    threshold = SLOWMO_RATIO * median

    low = [i for i, m in enumerate(motion) if 0 < m < threshold]
    if not low:
        return []

    grouped: list[tuple[int, int]] = []
    start = prev = low[0]
    for i in low[1:]:
        if i - prev <= SLOWMO_GAP_SEC:
            prev = i
        else:
            grouped.append((start, prev))
            start = prev = i
    grouped.append((start, prev))

    regions = [
        (float(a), float(b + 1))
        for a, b in grouped
        if (b - a + 1) >= SLOWMO_MIN_SEC
    ]

    flagged = sum(e - s for s, e in regions)
    if flagged > SLOWMO_MAX_FRAC * len(motion):
        return []  # détecteur peu fiable sur cette vidéo → on ne coupe rien
    return regions


def subtract_slowmo(
    start: float, end: float, regions: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Retire les zones de ralenti de l'intervalle [start, end] (peut le scinder)."""
    if not regions:
        return [(start, end)]
    segs = [(start, end)]
    for rs, re in regions:
        nxt: list[tuple[float, float]] = []
        for s, e in segs:
            if re <= s or rs >= e:
                nxt.append((s, e))
                continue
            if rs > s:
                nxt.append((s, min(rs, e)))
            if re < e:
                nxt.append((max(re, s), e))
        segs = nxt
    return [(s, e) for s, e in segs if e - s > 0.1]


def _select_clips(
    candidates: list[dict],
    target: float,
    slowmo_regions: list[tuple[float, float]],
    preselected: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    """Sélection gloutonne par score : retire les ralentis de chaque candidat puis
    place les sous-clips sans chevauchement jusqu'à atteindre la durée cible.

    `preselected` = clips déjà placés (les buts) que les extras ne doivent pas chevaucher
    et qui ne sont PAS renvoyés ici (seuls les extras sélectionnés le sont).

    Tri par PRIORITÉ croissante (but>penalty>...>ralenti) puis score décroissant : quand
    la cible est atteinte, ce sont les priorités faibles qui n'entrent pas."""
    ranked = sorted(
        candidates,
        key=lambda c: (PRIORITY.get(c.get("type", ""), 9), -c.get("score", 0.0)),
    )
    occupied: list[tuple[float, float]] = list(preselected or [])
    selected: list[dict] = []
    total = 0.0
    n_pre = len(preselected or [])
    for c in ranked:
        # Les candidats "replay" sont volontairement gardés tels quels (déjà plafonnés) ;
        # les autres sont nettoyés de tout ralenti.
        if c.get("keep_slowmo"):
            windows = [(c["start"], c["end"])]
        else:
            windows = subtract_slowmo(c["start"], c["end"], slowmo_regions)
        min_len = 1.0 if c.get("keep_slowmo") else MIN_CLIP_SEC
        for s, e in windows:
            if e - s < min_len:
                continue
            if any(not (e <= ss or s >= ee) for ss, ee in occupied):
                continue
            selected.append({"start": s, "end": e, "type": c.get("type", "")})
            occupied.append((s, e))
            total += e - s
            if len(selected) + n_pre >= MAX_CLIPS or total >= target:
                break
        if len(selected) + n_pre >= MAX_CLIPS or total >= target:
            break

    selected.sort(key=lambda d: d["start"])
    if total > target and selected:
        excess = total - target
        last = selected[-1]
        new_e = last["end"] - excess
        if new_e - last["start"] >= MIN_CLIP_SEC:
            last["end"] = new_e
        else:
            selected.pop()
    return selected


def trim_clips_slowmo(
    clips: list[tuple[float, float]], slowmo_regions: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Retire les ralentis d'une liste de clips existante (repli audio)."""
    if not slowmo_regions:
        return clips
    out: list[tuple[float, float]] = []
    for s, e in clips:
        out.extend(
            (a, b) for a, b in subtract_slowmo(s, e, slowmo_regions)
            if b - a >= MIN_CLIP_SEC
        )
    return out


def _dedupe_times(times: list[float], gap: float) -> list[float]:
    out: list[float] = []
    last = -10000.0
    for t in sorted(times):
        if t - last >= gap:
            out.append(t)
            last = t
    return out


def _ocr_digit_sequence(video: Path, x: float, prof: dict, step: float, tmp: Path, tag: str,
                        polarities=(("neg", ",negate"), ("pos", ""))) -> list[int | None]:
    """OCR d'un seul chiffre (domicile ou extérieur) toutes les `step` s. Par défaut teste
    les DEUX polarités (chiffre clair sur fond foncé ET foncé sur clair) ; passer
    `polarities=(("neg",",negate"),)` pour aller plus vite (probe). 0-9 ou None par échantillon."""
    crop = (f"fps=1/{step},crop=iw*{prof['digit_w']}:ih*{prof['digit_h']}:"
            f"iw*{x}:ih*{prof['digit_y']},scale=iw*10:ih*10:flags=lanczos,"
            f"format=gray,eq=contrast=1.6")
    out: dict[str, list[int | None]] = {}
    for pol, suffix in polarities:
        d = tmp / f"{tag}_{pol}"
        d.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video),
                        "-an", "-vf", crop + suffix, str(d / "f_%05d.png")],
                       capture_output=True, timeout=600)
        files = sorted(d.glob("f_*.png"))
        if not files:
            out[pol] = []
            continue
        lf = d / "list.txt"
        lf.write_text("\n".join(str(f) for f in files))
        r = subprocess.run(["tesseract", str(lf), "stdout", "--psm", "10",
                            "-c", "tessedit_char_whitelist=0123456789"],
                           capture_output=True, text=True, timeout=600)
        vals: list[int | None] = []
        for pg in r.stdout.split("\f"):
            s = pg.strip()
            vals.append(int(s) if (len(s) == 1 and s.isdigit()) else None)
        out[pol] = vals
    neg, pos = out.get("neg", []), out.get("pos", [])
    n = max(len(neg), len(pos))
    merged: list[int | None] = []
    for i in range(n):
        a = neg[i] if i < len(neg) else None
        b = pos[i] if i < len(pos) else None
        merged.append(a if a is not None else b)  # priorité polarité claire-sur-foncé (TF1)
    return merged


def auto_detect_scoreboard(video: Path, duration: float) -> dict:
    """AUTO-CALIBRATION (n'importe quelle chaîne) : trouve la position des chiffres du score.

    Scanne des positions candidates en haut de l'image (+ les profils connus). Pour chacune,
    OCR grossier des 2 chiffres sur ~14 images réparties. La VRAIE zone du score donne une
    valeur STABLE (le score reste constant longtemps) → on garde la position dont un total
    plausible (≤9) revient le plus souvent. Écarte l'horloge (qui change sans cesse) et la
    pelouse (illisible). Renvoie un profil utilisable par detect_score_changes."""
    from collections import Counter
    tmp = Path(tempfile.mkdtemp(prefix="auto_"))
    RW, RH = 0.34, 0.16          # zone haut-gauche extraite UNE seule fois (1 décodage)
    step = max(18.0, duration / 14)
    cands: list[dict] = [dict(SCOREBOARDS["tf1"]), dict(SCOREBOARDS["lequipe"])]
    for y in (0.05, 0.08, 0.11):
        for hx in (0.13, 0.17, 0.21, 0.25):
            cands.append({"home_x": hx, "away_x": hx + 0.030, "digit_y": y,
                          "digit_h": 0.055, "digit_w": 0.024,
                          "region_x": 0.0, "region_y": max(0.0, y - 0.03),
                          "region_w": 0.36, "region_h": 0.12, "score_delay": 8})

    def ocr_in_top(rel_x: float, rel_w: float, rel_y: float, rel_h: float, tag: str) -> list[int | None]:
        # recadre un chiffre DANS les frames déjà extraites (PNG → pas de re-décodage vidéo)
        d = tmp / tag
        d.mkdir(exist_ok=True)
        vf = (f"crop=iw*{rel_w}:ih*{rel_h}:iw*{rel_x}:ih*{rel_y},"
              f"scale=iw*10:ih*10:flags=lanczos,format=gray,negate,eq=contrast=1.6")
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                        "-i", str(tmp / "top" / "f_%04d.png"), "-vf", vf, str(d / "o_%04d.png")],
                       capture_output=True, timeout=300)
        files = sorted(d.glob("o_*.png"))
        if not files:
            return []
        lf = d / "l.txt"; lf.write_text("\n".join(str(f) for f in files))
        r = subprocess.run(["tesseract", str(lf), "stdout", "--psm", "10",
                            "-c", "tessedit_char_whitelist=0123456789"],
                           capture_output=True, text=True, timeout=300)
        return [int(s.strip()) if (len(s.strip()) == 1 and s.strip().isdigit()) else None
                for s in r.stdout.split("\f")]

    best, best_score = None, -1.0
    try:
        (tmp / "top").mkdir()
        subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
                        "-an", "-vf", f"fps=1/{step},crop=iw*{RW}:ih*{RH}:0:0",
                        str(tmp / "top" / "f_%04d.png")], capture_output=True, timeout=600)
        for i, prof in enumerate(cands):
            # positions du chiffre RELATIVES à la zone extraite (0-RW, 0-RH)
            h = ocr_in_top(prof["home_x"] / RW, prof["digit_w"] / RW,
                           prof["digit_y"] / RH, prof["digit_h"] / RH, f"h{i}")
            a = ocr_in_top(prof["away_x"] / RW, prof["digit_w"] / RW,
                           prof["digit_y"] / RH, prof["digit_h"] / RH, f"a{i}")
            totals = [hh + aa for hh, aa in zip(h, a)
                      if hh is not None and aa is not None and hh + aa <= 9]
            if not totals:
                continue
            top_total, freq = Counter(totals).most_common(1)[0]
            sc = freq + (0.5 if top_total > 0 else 0)  # stable + pas juste 0-0
            if sc > best_score:
                best_score, best = sc, prof
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not best:
        return SCOREBOARDS["tf1"]
    print(f"[auto-bandeau] position retenue: home_x={best['home_x']} away_x={best['away_x']} "
          f"y={best['digit_y']} (stabilité={best_score:.0f})", flush=True)
    return best


def detect_score_changes(video: Path, duration: float, channel: str = "tf1") -> list[float]:
    """SOURCE DE VÉRITÉ : OCR du SCORE → chaque incrémentation = un but (timing = transition).

    Lit les deux chiffres séparément (profil de chaîne pour la position), suit le score
    (total = domicile+extérieur) MONOTONE croissant. Un score n'est validé que s'il est
    STABLE (≥ SCORE_STABLE lectures) pour filtrer le bruit OCR / frames sans bandeau.
    Renvoie les instants de transition. Le nb de buts = score final (logué pour vérif)."""
    if shutil.which("tesseract") is None or duration <= 0:
        return []
    prof = (auto_detect_scoreboard(video, duration) if channel == "auto"
            else SCOREBOARDS.get(channel, SCOREBOARDS["tf1"]))
    tmp = Path(tempfile.mkdtemp(prefix="score_"))
    try:
        home = _ocr_digit_sequence(video, prof["home_x"], prof, SCORE_OCR_STEP, tmp, "h")
        away = _ocr_digit_sequence(video, prof["away_x"], prof, SCORE_OCR_STEP, tmp, "a")
        n = min(len(home), len(away))
        # lectures (temps, total) quand les DEUX chiffres sont lus
        reads = [(i * SCORE_OCR_STEP, home[i] + away[i])
                 for i in range(n) if home[i] is not None and away[i] is not None]
        if not reads:
            return []
        # un total n'est "stable" que s'il se répète sur SCORE_STABLE lectures proches
        stable: list[tuple[float, int]] = []
        for j, (t, tot) in enumerate(reads):
            agree = sum(1 for k in range(j, min(len(reads), j + 4)) if reads[k][1] == tot)
            if agree >= SCORE_STABLE:
                stable.append((t, tot))
        # détecte les incréments du score. UN BUT = +1 EXACTEMENT : on n'accepte qu'une
        # hausse d'une unité. Tout saut >1 (ex. 3→14 = OCR qui lit « 7 7 » hors bandeau)
        # ou toute baisse (vieux score d'un replay) est IGNORÉ → robuste au bruit OCR.
        goals: list[float] = []
        cur = None
        for t, tot in stable:
            if cur is None:
                cur = tot                 # score de départ (0-0 en général)
                continue
            if tot == cur + 1:            # +1 → un but ; on renvoie l'instant de TRANSITION
                goals.append(t)            # (l'ancrage sur l'action se fait ensuite via les ralentis)
                cur = tot
            # tot != cur+1 (saut OCR, baisse, replay) → ignoré ; cur reste inchangé
        print(f"[score] lectures={len(reads)} stables={len(stable)} score_final={cur} "
              f"buts={len(goals)} @ {[round(g) for g in goals]}", flush=True)
        return goals
    except (subprocess.TimeoutExpired, OSError):
        return []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def anchor_goals_on_action(transitions: list[float], regions: list[tuple[float, float]],
                           fallback_delay: float) -> list[float]:
    """Recale chaque but (instant de TRANSITION du score) sur son ACTION en direct.

    Le score se met à jour APRÈS célébration + RALENTI, dont la durée varie d'un but à
    l'autre → un délai fixe rate certains buts. Or le ralenti est en slow-motion (détecté
    dans `regions`) : l'action est juste AVANT le premier ralenti qui précède la maj du
    score. On ancre là. Repli sur transition − délai_chaîne si aucun ralenti trouvé."""
    out: list[float] = []
    for t in transitions:
        reps = sorted(r[0] for r in regions if t - 28 <= r[0] <= t - 1)
        if reps:
            # action ≈ qq s avant le début du replay (action → célébration → ralenti)
            out.append(max(0.0, reps[0] - 6.0))
        else:
            out.append(max(0.0, t - fallback_delay))
    return out


def detect_audio_roars(rms: list[float]) -> list[float]:
    """SIGNAL 3 : "roar" du public = montée sonore SOUTENUE (≈10-20 s) après un but,
    par opposition au pic bref d'une occasion. Renvoie les instants de début de roar."""
    n = len(rms)
    if n < 20:
        return []
    base = sorted(rms)[n // 2] or 1.0  # médiane
    loud = [1 if v > base * ROAR_FACTOR else 0 for v in rms]
    roars: list[float] = []
    i = 0
    while i < n - 4:
        window = loud[i:min(n, i + ROAR_WIN_SEC)]
        if sum(window) / len(window) >= ROAR_FRACTION:
            roars.append(float(i))
            i += ROAR_WIN_SEC  # on saute la durée du roar pour ne pas le recompter
        else:
            i += 1
    return roars


def fuse_goal_events(
    txt_goals: list[float], ocr_goals: list[float], roars: list[float],
    audio_peaks: list[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Fusionne les 3 signaux en clusters temporels et renvoie (buts_confirmés, faibles).

    Chaque cluster cumule la confiance des signaux présents. Un but est CONFIRMÉ s'il a
    l'OCR, ou (commentaire + roar), ou une confiance ≥ GOAL_CONFIRM. Les clusters trop
    faibles (un seul signal léger) sont renvoyés à part pour devenir de simples occasions."""
    # GARDE-FOU : un changement de score qui fire trop souvent = bandeau qui clignote
    # (montage), pas des buts → détecteur non fiable sur cette vidéo, on l'ignore.
    if len(ocr_goals) > MAX_PLAUSIBLE_GOALS:
        ocr_goals = []

    peaks = audio_peaks or []
    evidence = ([(t, W_OCR, "ocr") for t in ocr_goals]
                + [(t, W_TXT, "txt") for t in txt_goals]
                + [(t, W_ROAR, "roar") for t in roars])
    if not evidence:
        return [], []
    evidence.sort(key=lambda e: e[0])

    clusters: list[list[tuple[float, int, str]]] = []
    for ev in evidence:
        if clusters and ev[0] - clusters[-1][-1][0] <= GOAL_DEDUPE_SEC:
            clusters[-1].append(ev)
        else:
            clusters.append([ev])

    confirmed: list[float] = []
    weak: list[float] = []
    for cl in clusters:
        srcs = {s for _t, _w, s in cl}
        by_src = {s: t for t, _w, s in cl}
        t = by_src.get("ocr") or by_src.get("txt") or by_src.get("roar")
        # SCORE (ocr) = vérité terrain, JAMAIS écarté. Commentaire (txt) = fiable aussi.
        # Un roar SEUL (clameur sans score ni mot-clé) reste une simple occasion.
        if "ocr" in srcs or "txt" in srcs:
            confirmed.append(t)
        else:
            weak.append(t)
    return _dedupe_times(confirmed, GOAL_DEDUPE_SEC), weak


def build_clips_from_transcript(
    segments: list[dict], target: float, duration: float,
    slowmo_regions: list[tuple[float, float]] | None = None,
    rms: list[float] | None = None,
    ocr_goals: list[float] | None = None,
    sport: str = "foot",
) -> tuple[list[tuple[float, float]], dict, int]:
    """Construit le résumé en respectant la HIÉRARCHIE des événements.

    FOOT : buts/penalties/rouges OBLIGATOIRES (fusion commentaire+OCR+roar), reste par
    priorité. NBA : aucun obligatoire (on marque sans cesse) → meilleurs temps forts
    (dunk/3pts/contre/clutch + intensité audio) répartis sur tout le match.
    Clips triés chronologiquement pour préserver le récit."""
    regions = sorted(slowmo_regions or [])
    sorted_segs = sorted(segments, key=lambda s: s["start"])
    classify = classify_segment_nba if sport == "nba" else classify_segment

    # --- 1. Extraction typée des événements depuis le commentaire ---
    by_type: dict[str, list[dict]] = {
        "goal": [], "penalty": [], "redcard": [], "save": [], "chance": []
    }
    for seg in sorted_segs:
        typ, sc = classify(seg["text"])
        if typ:
            by_type[typ].append({"time": seg["start"], "score": sc})

    # FUSION : buts = commentaire (mots-clés) + changement de score (corroboré audio) + roar.
    txt_goals = _dedupe_times([e["time"] for e in by_type["goal"]], GOAL_DEDUPE_SEC)
    roars = detect_audio_roars(rms or [])
    peak_times = [(p["start"] + p["end"]) / 2 for p in audio_peak_candidates(rms or [], duration)]
    goal_starts, weak_goals = fuse_goal_events(txt_goals, ocr_goals or [], roars, peak_times)
    # un "but" douteux (un seul signal faible) devient une simple occasion, pas un but
    for t in weak_goals:
        by_type["chance"].append({"time": t, "score": 6.0})

    def near_goal(t: float, before: float = GOAL_PRE_SEC, after: float = ACTION_EXCLUDE_SEC) -> bool:
        return any(g - before <= t <= g + after for g in goal_starts)

    # penalties / rouges hors d'un but déjà détecté (un penalty marqué = déjà un but)
    penalty_starts = [t for t in _dedupe_times([e["time"] for e in by_type["penalty"]], GOAL_DEDUPE_SEC)
                      if not near_goal(t, before=10, after=10)]
    redcard_starts = _dedupe_times([e["time"] for e in by_type["redcard"]], GOAL_DEDUPE_SEC)

    # anchors obligatoires (type, temps)
    anchors = ([("goal", t) for t in goal_starts]
               + [("penalty", t) for t in penalty_starts]
               + [("redcard", t) for t in redcard_starts])
    anchors.sort(key=lambda a: a[1])
    n_goals = len(goal_starts)

    # --- 2. Candidats "extras" (célébrations, ralentis, occasions, arrêts, audio) ---
    late = duration * 0.85  # coup de pouce aux actions de fin de match (narration)
    extras: list[dict] = []

    def add(start: float, end: float, score: float, typ: str) -> None:
        boost = 1.3 if start >= late else 1.0
        extras.append({"start": start, "end": end, "score": score * boost, "type": typ})

    for g in goal_starts:  # célébration juste après le but
        cs = min(duration, g + GOAL_POST_SEC)
        ce = min(duration, cs + 4)
        if ce - cs >= 2:
            add(cs, ce, 5.0, "celebration")
    for g in goal_starts:  # un ralenti court par but
        rep = next((r for r in regions if g + GOAL_POST_SEC - 1 <= r[0] <= g + ACTION_EXCLUDE_SEC), None)
        if rep:
            rs, re = rep
            extras.append({"start": rs, "end": min(re, rs + SLOWMO_KEEP_SEC),
                           "score": 5.0, "type": "replay", "keep_slowmo": True})
    for typ in ("chance", "save"):  # occasions franches / arrêts décisifs
        for e in by_type[typ]:
            t = e["time"]
            if near_goal(t) or any(abs(t - at) < ACTION_EXCLUDE_SEC for _tp, at in anchors):
                continue
            add(max(0.0, t - ACTION_PRE_SEC), min(duration, t + ACTION_POST_SEC), e["score"], typ)
    for pc in audio_peak_candidates(rms or [], duration):  # clameur sans mot-clé
        c = (pc["start"] + pc["end"]) / 2
        if near_goal(c) or any(abs(c - at) < ACTION_EXCLUDE_SEC for _tp, at in anchors):
            continue
        add(pc["start"], pc["end"], pc["score"], "audio")

    # --- 3. COUVERTURE : garantir que TOUT le match est représenté (début → fin) ---
    # On découpe la durée en fenêtres temporelles ; toute fenêtre sans événement
    # obligatoire reçoit son meilleur candidat → le résumé s'étale sur l'intégralité
    # du match (c'est ce qui manquait : la fin n'était jamais couverte).
    n_windows = max(4, min(18, round(target / 8))) if duration > 0 else 1
    win = (duration / n_windows) if (duration > 0 and n_windows) else 1.0

    def widx(t: float) -> int:
        return min(n_windows - 1, int(t / win)) if win > 0 else 0

    anchor_windows = {widx(t) for _typ, t in anchors}
    ext_by_win: dict[int, list[dict]] = {}
    for c in extras:
        ext_by_win.setdefault(widx((c["start"] + c["end"]) / 2), []).append(c)

    last_w = n_windows - 1
    coverage: list[tuple[str, float]] = []
    for w in range(n_windows):
        if w in anchor_windows and w != last_w:
            continue
        cands = ext_by_win.get(w)
        if not cands:
            continue
        if w == last_w:
            # dernière fenêtre : on vise le moment le plus TARDIF (jusqu'au sifflet final)
            best = max(cands, key=lambda c: (c["start"] + c["end"]) / 2)
            ct = (best["start"] + best["end"]) / 2
            # ne pas dédoubler si un obligatoire couvre déjà la toute fin
            if not any(abs(ct - at) < GOAL_DEDUPE_SEC for _tp, at in anchors if at >= last_w * win):
                coverage.append((best["type"], ct))
        elif w not in anchor_windows:
            best = min(cands, key=lambda c: (PRIORITY.get(c["type"], 9), -c["score"]))
            coverage.append((best["type"], (best["start"] + best["end"]) / 2))

    # --- 4. Clips garantis : BUTS (généreux, biaisés AVANT) + couverture (courte) ---
    guaranteed: list[dict] = []
    placed: list[tuple[float, float]] = []

    def try_place(t: float, pre: float, post: float, typ: str, trim: bool = True) -> None:
        s, e = max(0.0, t - pre), min(duration, t + post)
        if trim:  # les occasions sont nettoyées des ralentis ; PAS les buts (on garde l'action)
            segs = subtract_slowmo(s, e, regions)
            if segs:
                s, e = max(segs, key=lambda x: x[1] - x[0])
        if e - s > 0.5 and not any(not (e <= ps or s >= pe) for ps, pe in placed):
            placed.append((s, e))
            guaranteed.append({"start": s, "end": e, "type": typ})

    # 4a. Buts : fenêtre large [transition−18, transition+3], NON rognée par les ralentis
    # (on garde l'action où qu'elle soit dans la fenêtre).
    for typ, t in sorted(anchors, key=lambda x: x[1]):
        try_place(t, GOAL_PRE_SEC, GOAL_POST_SEC, typ, trim=False)
    tot = sum(d["end"] - d["start"] for d in guaranteed)
    if tot > target and tot > 0:  # trop de buts pour la cible (ex. 30s) : on rétrécit
        scale = target / tot
        for d in guaranteed:
            # le but est près de la FIN (la transition de score) → on garde la fin
            new_len = (d["end"] - d["start"]) * scale
            d["start"] = max(0.0, d["end"] - new_len)
        placed = [(d["start"], d["end"]) for d in guaranteed]

    # PRIORITÉ AUX BUTS : si les buts remplissent déjà l'essentiel de la cible (match à
    # beaucoup de buts), on n'ajoute PAS d'occasions → on ne montre que les buts.
    goals_total = sum(d["end"] - d["start"] for d in guaranteed)
    allow_occasions = goals_total < GOALS_FILL_FRAC * target

    if allow_occasions:
        # 4b. Couverture (étalement sur tout le match) : clips COURTS d'occasions
        for typ, t in sorted(coverage, key=lambda x: x[1]):
            if sum(d["end"] - d["start"] for d in guaranteed) >= target:
                break
            try_place(t, ACTION_PRE_SEC, ACTION_POST_SEC, typ)

    used = sum(d["end"] - d["start"] for d in guaranteed)
    remaining = max(0.0, target - used)

    if not guaranteed and not extras:
        return [], {}, 0

    fill = (_select_clips(extras, remaining, regions,
                          preselected=[(d["start"], d["end"]) for d in guaranteed])
            if allow_occasions else [])
    all_dicts = sorted(guaranteed + fill, key=lambda d: d["start"])

    # counts = ce qui est RÉELLEMENT inclus (pas les candidats détectés)
    counts = {k: 0 for k in ("goal", "penalty", "redcard", "chance", "save", "audio", "celebration", "replay")}
    for d in all_dicts:
        counts[d.get("type", "")] = counts.get(d.get("type", ""), 0) + 1

    clips = [(d["start"], d["end"]) for d in all_dicts]
    return clips, counts, n_goals

app = Flask(__name__)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def update_job(job_id: str, **kwargs) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)


def safe_filename(name: str, max_len: int = 60) -> str:
    cleaned = re.sub(r"[^\w\s-]", "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:max_len] or "video"


def get_video_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def extract_audio_rms(path: Path) -> list[float] | None:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(AUDIO_SR),
        "-f", "s16le", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=180)
    if result.returncode != 0 or not result.stdout:
        return None

    samples = array.array("h")
    samples.frombytes(result.stdout)
    if not samples:
        return None

    rms_per_sec: list[float] = []
    for i in range(0, len(samples) - AUDIO_SR + 1, AUDIO_SR):
        chunk = samples[i:i + AUDIO_SR]
        s = sum(x * x for x in chunk) / len(chunk)
        rms_per_sec.append(s ** 0.5)
    return rms_per_sec


def find_highlight_clips(rms: list[float], target: float, duration: float) -> list[tuple[float, float]]:
    n = len(rms)
    if n == 0 or duration <= 0:
        return []
    if duration <= target:
        return [(0.0, duration)]

    smoothed = []
    for i in range(n):
        s = max(0, i - 1)
        e = min(n, i + 2)
        smoothed.append(sum(rms[s:e]) / (e - s))

    sorted_vals = sorted(smoothed)
    threshold = sorted_vals[min(n - 1, int(n * PEAK_PERCENTILE))]
    above = [i for i, v in enumerate(smoothed) if v >= threshold]

    if not above:
        ranked = sorted(range(n), key=lambda i: smoothed[i], reverse=True)
        above = sorted(ranked[: max(3, MAX_CLIPS)])

    groups: list[dict] = []
    current = [above[0]]
    for i in above[1:]:
        if i - current[-1] <= GROUP_GAP_SEC:
            current.append(i)
        else:
            peak = max(current, key=lambda x: smoothed[x])
            groups.append({"peak": peak, "score": smoothed[peak]})
            current = [i]
    peak = max(current, key=lambda x: smoothed[x])
    groups.append({"peak": peak, "score": smoothed[peak]})

    groups.sort(key=lambda g: g["score"], reverse=True)

    selected: list[tuple[float, float]] = []
    total = 0.0
    for g in groups[: MAX_CLIPS * 2]:
        peak = g["peak"]
        start = max(0.0, peak - PRE_PEAK_SEC)
        end = min(duration, peak + POST_PEAK_SEC)
        if end - start < MIN_CLIP_SEC:
            continue
        if any(not (end <= s or start >= e) for s, e in selected):
            continue
        selected.append((start, end))
        total += end - start
        if len(selected) >= MAX_CLIPS or total >= target:
            break

    if not selected:
        return [(0.0, min(target, duration))]

    selected.sort()
    total = sum(e - s for s, e in selected)
    if total > target:
        excess = total - target
        last_start, last_end = selected[-1]
        new_end = last_end - excess
        if new_end - last_start >= MIN_CLIP_SEC:
            selected[-1] = (last_start, new_end)
        else:
            selected.pop()

    return selected


def audio_peak_candidates(rms: list[float], duration: float) -> list[dict]:
    """Candidats "occasion" basés sur les pics de bruit (clameur du public/commentateur).

    Complète le commentaire : capte les temps forts SANS mot-clé (arrêts, occasions,
    voire des buts ratés par la transcription) et donne assez de matière pour remplir
    la durée demandée. Score normalisé ~[2, 12] pour s'interclasser avec les actions."""
    n = len(rms)
    if n == 0 or duration <= 0:
        return []
    smoothed = []
    for i in range(n):
        s = max(0, i - 1)
        e = min(n, i + 2)
        smoothed.append(sum(rms[s:e]) / (e - s))

    sorted_vals = sorted(smoothed)
    threshold = sorted_vals[min(n - 1, int(n * PEAK_PERCENTILE))]
    above = [i for i, v in enumerate(smoothed) if v >= threshold]
    if not above:
        return []

    groups: list[dict] = []
    current = [above[0]]
    for i in above[1:]:
        if i - current[-1] <= GROUP_GAP_SEC:
            current.append(i)
        else:
            peak = max(current, key=lambda x: smoothed[x])
            groups.append({"peak": peak, "score": smoothed[peak]})
            current = [i]
    peak = max(current, key=lambda x: smoothed[x])
    groups.append({"peak": peak, "score": smoothed[peak]})
    if not groups:
        return []

    lo = min(g["score"] for g in groups)
    span = (max(g["score"] for g in groups) - lo) or 1.0
    cands: list[dict] = []
    for g in groups:
        pk = g["peak"]
        cands.append({
            "start": max(0.0, pk - PRE_PEAK_SEC),
            "end": min(duration, pk + POST_PEAK_SEC),
            "score": 2.0 + 10.0 * (g["score"] - lo) / span,
            "type": "audio",
        })
    return cands


LOGO_REGION_W = 0.18
LOGO_REGION_H = 0.13
LOGO_REGION_X = 0.82
LOGO_BLUR_STRENGTH = "boxblur=20:3"

# Tableau des scores (par défaut en haut À GAUCHE) → recollé en haut-centre.
# Calé sur le bandeau TF1 (FRA 1-0 CIV). Réglable selon le diffuseur.
SCORE_REGION_X = 0.06
SCORE_REGION_Y = 0.04
SCORE_REGION_W = 0.28
SCORE_REGION_H = 0.09
SCORE_OUT_W = 620       # largeur du bandeau recollé (sur 1080 de large)
SCORE_MARGIN_TOP = 90   # marge depuis le haut (descendu pour ne pas être trop haut)

# Suivi de l'action (mode recadré) : panoramique vers la zone de plus fort mouvement.
PAN_FPS = 5             # échantillonnage de la trajectoire
PAN_GRID_W = 32         # colonnes d'analyse du mouvement
PAN_GRID_H = 18
PAN_SMOOTH = 0.20       # lissage EMA (petit = plus lisse / plus lent)
PAN_MIN_STEP_PX = 6     # on ne crée un point de trajectoire que si ça bouge d'au moins ça
MAX_PAN_POINTS = 24     # plafond de points (sinon l'expression ffmpeg casse)
PAN_FOLLOW = False      # False = recadré CENTRÉ (la caméra TV suit déjà le ballon, meilleur
                        # au foot) ; True = panoramique par mouvement (2 passes, expérimental)


def compute_pan_expr(work: Path, W: int, H: int) -> str | None:
    """Trajectoire de panoramique horizontale qui suit la zone de plus fort mouvement.

    Une passe ffmpeg sort une mini-vidéo de différence (32×18, niveaux de gris) en brut ;
    pour chaque image on calcule le centroïde horizontal du mouvement, on lisse (EMA),
    puis on en fait une expression x(t) pour le filtre crop (pas de sendcmd → pas de
    conflit avec les autres crops). Renvoie None si la source est déjà ~verticale."""
    cropw = round(H * 9 / 16)
    max_x = W - cropw
    if max_x <= 1:
        return None  # déjà au format ~9:16, rien à suivre

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(work),
        "-an", "-vf",
        f"fps={PAN_FPS},tblend=all_mode=difference,scale={PAN_GRID_W}:{PAN_GRID_H},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0 or not result.stdout:
        return None

    frame_size = PAN_GRID_W * PAN_GRID_H
    data = result.stdout
    n_frames = len(data) // frame_size
    if n_frames < 2:
        return None

    ema = (max_x / 2)  # départ au centre
    pts: list[tuple[float, float]] = []
    for f in range(n_frames):
        frame = data[f * frame_size:(f + 1) * frame_size]
        col = [0] * PAN_GRID_W
        for row in range(PAN_GRID_H):
            base = row * PAN_GRID_W
            for c in range(PAN_GRID_W):
                col[c] += frame[base + c]
        # pondération au CARRÉ : focalise sur la zone de mouvement la plus intense
        # (l'action/le ballon) plutôt que la moyenne tirée vers le centre par la foule
        w2 = [c * c for c in col]
        tot = sum(w2)
        if tot > 0:
            centroid = sum(i * w2[i] for i in range(PAN_GRID_W)) / tot   # 0..GRID_W-1
            cx = centroid / (PAN_GRID_W - 1)                              # 0..1
            target = min(max_x, max(0.0, cx * W - cropw / 2))
            ema += PAN_SMOOTH * (target - ema)                           # lissage
        t = f / PAN_FPS
        pts.append((t, ema))

    # décimation : on ne garde un point que s'il bouge assez (expression plus légère)
    kept: list[tuple[float, float]] = [pts[0]]
    for t, x in pts[1:]:
        if abs(x - kept[-1][1]) >= PAN_MIN_STEP_PX or t - kept[-1][0] >= 4:
            kept.append((t, x))
    if kept[-1][0] < pts[-1][0]:
        kept.append(pts[-1])

    # PLAFOND : ré-échantillonne uniformément pour rester sous la limite ffmpeg
    # (l'évaluateur d'expressions casse au-delà d'une certaine profondeur/longueur).
    if len(kept) > MAX_PAN_POINTS:
        idx = sorted({round(i * (len(kept) - 1) / (MAX_PAN_POINTS - 1))
                      for i in range(MAX_PAN_POINTS)})
        kept = [kept[i] for i in idx]

    # Expression x(t) linéaire par morceaux À PLAT (somme de segments disjoints,
    # PAS de if() imbriqués → évite "too many args" de l'évaluateur ffmpeg).
    terms = [f"lt(t,{kept[0][0]:.3f})*{kept[0][1]:.1f}"]
    for i in range(len(kept) - 1):
        t0, x0 = kept[i]
        t1, x1 = kept[i + 1]
        dt = max(0.001, t1 - t0)
        slope = (x1 - x0) / dt
        terms.append(f"gte(t,{t0:.3f})*lt(t,{t1:.3f})*({x0:.1f}+{slope:.3f}*(t-{t0:.3f}))")
    terms.append(f"gte(t,{kept[-1][0]:.3f})*{kept[-1][1]:.1f}")
    return "+".join(terms)


def probe_dimensions(path: Path) -> tuple[int, int]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    try:
        w, h = out.stdout.strip().split("x")
        return int(w), int(h)
    except (ValueError, AttributeError):
        return 0, 0


def add_logo_blur(label_in: str, label_out: str) -> list[str]:
    return [
        f"[{label_in}]split=2[{label_in}_o][{label_in}_l]",
        f"[{label_in}_l]crop=iw*{LOGO_REGION_W}:ih*{LOGO_REGION_H}:"
        f"iw*{LOGO_REGION_X}:0,{LOGO_BLUR_STRENGTH}[{label_in}_b]",
        f"[{label_in}_o][{label_in}_b]overlay=W*{LOGO_REGION_X}:0[{label_out}]",
    ]


def vertical_filter(
    in_label: str, style: str, blur_logo: bool, scoreboard: bool,
    pan_expr: str | None = None, region: tuple[float, float, float, float] | None = None,
) -> tuple[list[str], str]:
    """Construit le filtre vertical 1080×1920 : style + flou logo + (panoramique) + score.

    `pan_expr` (mode recadré) : expression x(t) pour suivre l'action ; None = centré.
    `scoreboard` : recolle la zone du score (selon `region`=(x,y,w,h) de la chaîne) en haut-centre."""
    rx, ry, rw, rh = region or (SCORE_REGION_X, SCORE_REGION_Y, SCORE_REGION_W, SCORE_REGION_H)
    parts: list[str] = []
    if scoreboard:
        parts.append(f"[{in_label}]split=2[vmain][vscore]")
        main = "vmain"
    else:
        main = in_label

    if blur_logo:
        parts.extend(add_logo_blur(main, "vlogo"))
        main = "vlogo"

    if style == "crop":
        xexpr = pan_expr if pan_expr else "(iw-ow)/2"  # suivi de l'action sinon centré
        parts.append(f"[{main}]crop=ih*9/16:ih:x='{xexpr}':y=0,scale=1080:1920[vbase]")
    elif style == "pad":
        parts.append(
            f"[{main}]scale=1080:1920:force_original_aspect_ratio=decrease,"
            f"pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black[vbase]"
        )
    else:  # blur
        parts.append(f"[{main}]split=2[bg_src][fg_src]")
        parts.append("[bg_src]scale=1080:1920:force_original_aspect_ratio=increase,"
                     "crop=1080:1920,gblur=sigma=30[bg]")
        parts.append("[fg_src]scale=1080:-2[fg]")
        parts.append("[bg][fg]overlay=(W-w)/2:(H-h)/2[vbase]")

    if scoreboard:
        parts.append(
            f"[vscore]crop=iw*{rw}:ih*{rh}:iw*{rx}:ih*{ry},scale={SCORE_OUT_W}:-2[score]"
        )
        parts.append(f"[vbase][score]overlay=(W-w)/2:{SCORE_MARGIN_TOP}[outv]")
        return parts, "[outv]"
    return parts, "[vbase]"


_ENC = [
    "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
]


def build_ffmpeg_manual(input_path: Path, output_path: Path, start: int, duration: int | None,
                        style: str, blur_logo: bool, scoreboard: bool = False, region=None) -> list[str]:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(input_path)]
    if duration:
        cmd += ["-t", str(duration)]
    parts, vmap = vertical_filter("0:v", style, blur_logo, scoreboard, region=region)
    cmd += ["-filter_complex", ";".join(parts), "-map", vmap, "-map", "0:a?"]
    cmd += [*_ENC, str(output_path)]
    return cmd


def build_ffmpeg_highlights(input_path: Path, output_path: Path, clips: list[tuple[float, float]],
                            style: str, blur_logo: bool, scoreboard: bool = False, region=None) -> list[str]:
    parts: list[str] = []
    for i, (s, e) in enumerate(clips):
        parts.append(f"[0:v]trim={s:.2f}:{e:.2f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim={s:.2f}:{e:.2f},asetpts=PTS-STARTPTS[a{i}]")
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(clips)))
    parts.append(f"{concat_inputs}concat=n={len(clips)}:v=1:a=1[vc][ac]")
    vparts, vmap = vertical_filter("vc", style, blur_logo, scoreboard, region=region)
    parts.extend(vparts)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path)]
    cmd += ["-filter_complex", ";".join(parts), "-map", vmap, "-map", "[ac]"]
    cmd += [*_ENC, str(output_path)]
    return cmd


def build_assemble_highlights(input_path: Path, work_path: Path,
                              clips: list[tuple[float, float]]) -> list[str]:
    """Assemble (sans format vertical) les extraits en une vidéo horizontale haute qualité."""
    parts: list[str] = []
    for i, (s, e) in enumerate(clips):
        parts.append(f"[0:v]trim={s:.2f}:{e:.2f},setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim={s:.2f}:{e:.2f},asetpts=PTS-STARTPTS[a{i}]")
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(len(clips)))
    parts.append(f"{concat_inputs}concat=n={len(clips)}:v=1:a=1[vc][ac]")
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path),
            "-filter_complex", ";".join(parts), "-map", "[vc]", "-map", "[ac]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "160k", str(work_path)]


def build_assemble_manual(input_path: Path, work_path: Path, start: int,
                          duration: int | None) -> list[str]:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(input_path)]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-b:a", "160k", str(work_path)]
    return cmd


COLOR_LOOKS = {
    "vif": "eq=contrast=1.08:saturation=1.30:brightness=0.01",
    "chaud": "colorbalance=rs=0.06:rm=0.04:bs=-0.06:bm=-0.03",
    "froid": "colorbalance=rs=-0.05:bs=0.07:bm=0.03",
    "cinema": "eq=contrast=1.06:saturation=0.95,"
              "colorbalance=rs=-0.04:bs=0.06:rm=0.04:bm=-0.02",
}


def build_personalization(in_path: Path, out_path: Path, speed: float,
                          look: str, mute: bool) -> list[str]:
    """Passe créative finale : vitesse (rythme), look couleur, son (couper).

    Vitesse : `setpts` (vidéo) + `atempo` (audio) au MÊME facteur. Pour éviter toute
    dérive A/V (sources YouTube souvent en fréquence d'images variable), on force la
    sortie en fréquence constante et on resynchronise l'audio."""
    chg = abs(speed - 1.0) > 0.001
    vf: list[str] = []
    if look in COLOR_LOOKS:
        vf.append(COLOR_LOOKS[look])
    if chg:
        vf.append(f"setpts=PTS/{speed:.3f}")

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(in_path)]
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if mute:
        cmd += ["-an"]
    elif chg:
        # atempo + resynchro audio (aresample async) → reste calé sur la vidéo
        cmd += ["-af", f"atempo={speed:.3f},aresample=async=1:first_pts=0"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p"]
    if chg:
        # fréquence d'images CONSTANTE → pas de dérive vidéo sur source VFR
        cmd += ["-fps_mode", "cfr", "-r", "30"]
    cmd += ["-c:a", "aac", "-b:a", "128k"] if not mute else []
    cmd += ["-movflags", "+faststart", str(out_path)]
    return cmd


def build_ffmpeg_style_pass(work_path: Path, output_path: Path, style: str,
                            blur_logo: bool, scoreboard: bool, pan_expr: str | None, region=None) -> list[str]:
    """Passe 2 (mode recadré) : applique le format vertical avec panoramique + score."""
    parts, vmap = vertical_filter("0:v", style, blur_logo, scoreboard, pan_expr, region)
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(work_path),
            "-filter_complex", ";".join(parts), "-map", vmap, "-map", "0:a?",
            *_ENC, str(output_path)]


def render_vertical(job_id: str, download_path: Path, output_path: Path, mode: str, style: str,
                    blur_logo: bool, scoreboard: bool, clips=None, start: int = 0,
                    duration_opt: int | None = None, region=None) -> None:
    """Rendu final. Recadré = centré (la caméra TV suit déjà le ballon ; le panoramique
    par mouvement dégradait l'image au foot). PAN_FOLLOW=True réactive le suivi (2 passes)."""
    print(f"[{job_id[:8]}] rendu style={style} score={scoreboard} flou_logo={blur_logo} mode={mode}", flush=True)
    work_path = DOWNLOADS_DIR / f"{job_id}_work.mp4"
    try:
        if style == "crop" and PAN_FOLLOW:
            if mode == "auto":
                acmd = build_assemble_highlights(download_path, work_path, clips)
            else:
                acmd = build_assemble_manual(download_path, work_path, start, duration_opt)
            r = subprocess.run(acmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg(assemblage): {r.stderr.strip()[:400]}")
            update_job(job_id, progress=93)
            w, h = probe_dimensions(work_path)
            pan_expr = compute_pan_expr(work_path, w, h) if (w and h) else None
            print(f"[{job_id[:8]}] recadré {w}x{h} suivi_action={'oui' if pan_expr else 'non'}", flush=True)
            cmd = build_ffmpeg_style_pass(work_path, output_path, style, blur_logo, scoreboard, pan_expr, region)
        elif mode == "auto":
            cmd = build_ffmpeg_highlights(download_path, output_path, clips, style, blur_logo, scoreboard, region)
        else:
            cmd = build_ffmpeg_manual(download_path, output_path, start, duration_opt, style, blur_logo, scoreboard, region)

        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg: {r.stderr.strip()[:500]}")
    finally:
        if work_path.exists():
            try:
                work_path.unlink()
            except OSError:
                pass


def process_video(job_id: str, url: str, options: dict) -> None:
    download_path = DOWNLOADS_DIR / f"{job_id}.mp4"
    output_path = OUTPUTS_DIR / f"{job_id}.mp4"
    try:
        update_job(job_id, status="downloading", progress=5)

        def hook(d):
            if d["status"] == "downloading":
                total = d.get("total_bytes") or d.get("total_bytes_estimate")
                if total:
                    pct = int(d.get("downloaded_bytes", 0) / total * 40) + 5
                    update_job(job_id, progress=min(pct, 45))

        ydl_opts = {
            # souple : mp4/h264 si dispo, sinon VP9/webm (fusionné + ré-encodé ensuite)
            "format": "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/"
                      "bv*[height<=1080]+ba/b[height<=1080]/best",
            "outtmpl": str(download_path.with_suffix(".%(ext)s")),
            "merge_output_format": "mp4",
            "quiet": True,
            "noplaylist": True,
            "progress_hooks": [hook],
        }
        # Cookies du navigateur → contourne l'anti-bot YouTube ("Sign in to confirm…")
        # Défaut adapté à l'OS : Safari (macOS uniquement), Edge (Windows), sinon Firefox.
        default_browser = "safari" if sys.platform == "darwin" else (
            "edge" if sys.platform == "win32" else "firefox"
        )
        browser = os.environ.get("COOKIES_BROWSER", default_browser).strip().lower()

        def _download(use_cookies):
            opts = dict(ydl_opts)
            if use_cookies and browser and browser != "none":
                opts["cookiesfrombrowser"] = (browser,)
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return info.get("title") or "video"

        try:
            title = _download(use_cookies=True)
        except Exception as e:
            msg = str(e)
            # Toute erreur liée aux cookies (plateforme non supportée, base verrouillée/chiffrée,
            # impossible à copier…) → on retente sans cookies plutôt que de planter.
            cookie_problem = (
                "unsupported platform" in msg.lower()
                or "cookie" in msg.lower()
            )
            if cookie_problem:
                # Lecture des cookies impossible (ex. Safari sur Windows) → on réessaie sans cookies.
                try:
                    title = _download(use_cookies=False)
                except Exception as e2:
                    msg2 = str(e2)
                    if "sign in to confirm" in msg2.lower() or "not a bot" in msg2.lower():
                        raise RuntimeError(
                            "YouTube bloque (anti-bot) et les cookies du navigateur sont "
                            f"illisibles sur ce système. Définis COOKIES_BROWSER sur un "
                            f"navigateur installé (chrome, edge, firefox) où tu es connecté à "
                            f"YouTube. Détail: {msg2[:160]}"
                        )
                    raise
            elif "sign in to confirm" in msg.lower() or "not a bot" in msg.lower():
                raise RuntimeError(
                    f"YouTube bloque (anti-bot) malgré les cookies {browser}. Ouvre la vidéo "
                    f"dans {browser} en étant connecté, puis réessaie. Détail: {msg[:160]}"
                )
            else:
                raise

        produced = next(DOWNLOADS_DIR.glob(f"{job_id}.*"), None)
        if produced is None or not produced.exists():
            raise RuntimeError("Téléchargement YouTube échoué")
        if produced.suffix != ".mp4":
            produced.replace(download_path)
        else:
            download_path = produced

        mode = options.get("mode", "auto")
        style = options.get("format", "blur")
        blur_logo = bool(options.get("blur_logo", True))
        cut_slowmo = bool(options.get("cut_slowmo", True))
        scoreboard = bool(options.get("scoreboard", False))
        speed = float(options.get("speed", 1.0))
        look = options.get("look", "none")
        mute = bool(options.get("mute", False))
        sport = options.get("sport", "foot")
        channel = options.get("channel", "auto")
        if channel in SCOREBOARDS:
            prof = SCOREBOARDS[channel]
            score_region = (prof["region_x"], prof["region_y"], prof["region_w"], prof["region_h"])
        else:  # auto : région overlay générique haut-gauche
            score_region = (0.0, 0.04, 0.36, 0.12)

        if mode == "auto":
            target = float(options.get("target_duration") or 60)
            duration = get_video_duration(download_path)

            slowmo_regions: list[tuple[float, float]] = []
            if cut_slowmo and duration > 0:
                update_job(job_id, status="detecting", progress=47, title=title)
                motion = compute_motion_per_sec(download_path, duration)
                if motion:
                    slowmo_regions = detect_slowmo_regions(motion)
                sm_total = sum(e - s for s, e in slowmo_regions)
                print(
                    f"[{job_id[:8]}] durée={duration:.0f}s cible={target:.0f}s "
                    f"motion={'OK' if motion else 'ÉCHEC'} "
                    f"ralentis={len(slowmo_regions)} ({sm_total:.0f}s): "
                    f"{[(round(s), round(e)) for s, e in slowmo_regions]}",
                    flush=True,
                )

            # Pics audio = source d'"occasions" (et filet pour buts ratés par le texte)
            rms = extract_audio_rms(download_path)

            # OCR du score : utile seulement au FOOT (buts rares). En NBA on marque sans
            # cesse → inutile pour sélectionner ; on s'appuie sur l'audio + mots-clés.
            ocr_goals = []
            if sport != "nba" and duration > 0:
                update_job(job_id, status="reading_score", progress=49)
                # instants de TRANSITION du score (l'action est avant ; le clip englobe les
                # deux : on remonte large ET on finit après la transition → capte les buts
                # à délai court comme à délai long, sans avoir à deviner le délai exact)
                ocr_goals = detect_score_changes(download_path, duration, channel)
                print(f"[{job_id[:8]}] score ({channel}): {len(ocr_goals)} but(s) -> "
                      f"{[round(t) for t in ocr_goals]}", flush=True)

            update_job(job_id, status="transcribing", progress=50, title=title)
            clips: list[tuple[float, float]] = []
            detection = ""
            try:
                # langue : foot = fr (TF1) ; nba = auto (souvent EN). Env > sport.
                lang = None if sport == "nba" else "fr"
                segments = transcribe_video(download_path, job_id, duration, language=lang)
                clips, counts, n_goals = build_clips_from_transcript(
                    segments, target, duration, slowmo_regions, rms, ocr_goals, sport
                )
                print(
                    f"[{job_id[:8]}] clips={len(clips)} "
                    f"total={sum(e - s for s, e in clips):.0f}s/{target:.0f}s "
                    f"fin={max((e for _, e in clips), default=0):.0f}s "
                    f"events={counts} -> {[(round(s), round(e)) for s, e in clips]}",
                    flush=True,
                )
                if clips and sport == "nba":
                    detection = f"{len(clips)} temps forts (dunks, 3pts, contres, clutch…)"
                elif clips:
                    parts = [f"{counts['goal']} but(s)"]
                    if counts["penalty"]:
                        parts.append(f"{counts['penalty']} penalty")
                    if counts["redcard"]:
                        parts.append(f"{counts['redcard']} rouge(s)")
                    if counts["save"]:
                        parts.append(f"{counts['save']} arrêt(s)")
                    if counts["chance"] + counts["audio"]:
                        parts.append(f"{counts['chance'] + counts['audio']} occasion(s)")
                    detection = f"{len(clips)} clips — " + ", ".join(parts)
            except Exception as e:
                detection = f"Whisper indisponible ({type(e).__name__}), repli audio"
                print(f"[{job_id[:8]}] whisper KO: {type(e).__name__}: {e}", flush=True)

            if not clips:
                update_job(job_id, status="analyzing", progress=85)
                if rms and duration > 0:
                    clips = find_highlight_clips(rms, target, duration)
                    clips = trim_clips_slowmo(clips, slowmo_regions)
                    if clips and not detection.startswith("Whisper"):
                        detection = f"{len(clips)} moment(s) (repli sur pics audio)"
                if not clips:
                    clips = [(0.0, min(target, duration or target))]
                    detection = detection or "fallback début de vidéo"

            if slowmo_regions:
                detection = (detection or "résumé") + (
                    f" • ralentis réduits (1 court replay/but gardé, ≤{SLOWMO_KEEP_SEC}s)"
                )

            update_job(
                job_id,
                status="processing",
                progress=88,
                clips=[{"start": round(s, 1), "end": round(e, 1)} for s, e in clips],
                detection=detection,
                total_clip_duration=round(sum(e - s for s, e in clips), 1),
            )
            render_vertical(job_id, download_path, output_path, "auto", style,
                            blur_logo, scoreboard, clips=clips, region=score_region)
        else:
            update_job(job_id, status="processing", progress=55, title=title)
            start = int(options.get("start") or 0)
            duration_opt = options.get("duration")
            render_vertical(job_id, download_path, output_path, "manual", style,
                            blur_logo, scoreboard, start=start,
                            duration_opt=int(duration_opt) if duration_opt else None,
                            region=score_region)

        # Personnalisation créative (vitesse / look couleur / son) en passe finale
        if abs(speed - 1.0) > 0.001 or look not in ("none", "", None) or mute:
            tmp_perso = OUTPUTS_DIR / f"{job_id}_perso.mp4"
            cmd = build_personalization(output_path, tmp_perso, speed, look, mute)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0:
                tmp_perso.replace(output_path)
            else:
                print(f"[{job_id[:8]}] perso échec: {r.stderr.strip()[:200]}", flush=True)
                tmp_perso.unlink(missing_ok=True)

        update_job(
            job_id,
            status="done",
            progress=100,
            output_path=str(output_path),
            download_name=f"{safe_filename(title)}_tiktok.mp4",
        )
    except Exception as e:
        update_job(job_id, status="error", error=str(e))
    finally:
        if download_path.exists():
            try:
                download_path.unlink()
            except OSError:
                pass


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/process")
def submit():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "URL manquante"}), 400

    mode = data.get("mode", "auto")
    style = data.get("format", "blur")
    if style not in {"blur", "crop", "pad"}:
        style = "blur"
    if mode not in {"auto", "manual"}:
        mode = "auto"

    options = {
        "mode": mode,
        "format": style,
        "blur_logo": bool(data.get("blur_logo", True)),
        "cut_slowmo": bool(data.get("cut_slowmo", True)),
        "scoreboard": bool(data.get("scoreboard", False)),
        "mute": bool(data.get("mute", False)),
        "sport": data.get("sport") if data.get("sport") in ("foot", "nba") else "foot",
        "channel": data.get("channel") if data.get("channel") in (*SCOREBOARDS, "auto") else "auto",
    }
    try:
        sp = float(data.get("speed", 1.0))
        options["speed"] = sp if sp in (0.9, 0.95, 1.0, 1.05, 1.1) else 1.0
    except (ValueError, TypeError):
        options["speed"] = 1.0
    lk = data.get("look", "none")
    options["look"] = lk if lk in ("none", "vif", "chaud", "froid", "cinema") else "none"
    try:
        if mode == "auto":
            target_raw = data.get("target_duration") or 60
            options["target_duration"] = max(10, min(TARGET_MAX_SEC, int(target_raw)))
        else:
            options["start"] = int(data.get("start") or 0)
            d = data.get("duration")
            options["duration"] = int(d) if d not in (None, "", 0, "0") else None
    except ValueError:
        return jsonify({"error": "Valeurs numériques invalides"}), 400

    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"status": "pending", "progress": 0}

    threading.Thread(
        target=process_video, args=(job_id, url, options), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.get("/status/<job_id>")
def status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job inconnu"}), 404
    return jsonify(job)


@app.get("/download/<job_id>")
def download(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Vidéo pas prête"}), 404
    return send_file(
        job["output_path"], as_attachment=True,
        download_name=job.get("download_name", "tiktok.mp4"),
        mimetype="video/mp4",
    )


@app.get("/preview/<job_id>")
def preview(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Vidéo pas prête"}), 404
    return send_file(job["output_path"], mimetype="video/mp4")


if __name__ == "__main__":
    import os
    if shutil.which("ffmpeg") is None:
        print("⚠️  ffmpeg introuvable dans le PATH. Lance via run.ps1.")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
