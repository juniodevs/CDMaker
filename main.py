#!/usr/bin/env python3
"""
CDMaker — Now Playing Music Video Generator  (pipeline otimizado)
==================================================================
Uso
---
  python main.py cover.jpg musica.mp3 --handle NESTALGIA
  python main.py cover.jpg musica.mp3 --rpm 45 --fps 24 --preset ultrafast
  python main.py --batch ./musicas --handle NESTALGIA
"""

import os
import sys
import shutil
import subprocess
import textwrap
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np

try:
    from PIL import Image, ImageFilter, ImageDraw, ImageFont, ImageEnhance
except ImportError:
    sys.exit("Pillow não instalado.  Execute: pip install -r requirements.txt")

try:
    import librosa
except ImportError:
    sys.exit("librosa não instalado.  Execute: pip install -r requirements.txt")

try:
    from scipy.ndimage import uniform_filter1d
except ImportError:
    sys.exit("scipy não instalado.  Execute: pip install -r requirements.txt")

try:
    from mutagen import File as MutagenFile
except ImportError:
    MutagenFile = None


# ── FFmpeg ───────────────────────────────────────────────────────────────────

def _find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    sys.exit("ffmpeg não encontrado. Instale ffmpeg ou: pip install imageio-ffmpeg")

FFMPEG = _find_ffmpeg()

# ── Configurações ────────────────────────────────────────────────────────────

VIDEO_W  = 1920
VIDEO_H  = 1080
VIDEO_V_W = 1080
VIDEO_V_H = 1920
DEF_FPS     = 30
DEF_RPM     = 8.0    # RPM padrão visual (~7.5s por volta — suave para "now playing")
DEF_PRESET  = "fast"
DEF_WORKERS = 0      # 0 = auto (cpu_count)
N_BARS   = 200

# Cores (R, G, B, A)
C_TITLE   = (255, 255, 255, 255)
C_ARTIST  = (210, 210, 210, 255)
C_ALBUM   = (145, 145, 145, 255)
C_HANDLE  = (185, 185, 185, 210)
C_PLAYED  = (255, 255, 255, 255)
C_UNPLAY  = (72,  72,  72,  255)
C_CURSOR    = np.array([255.0, 205.0, 50.0], dtype=np.float32)   # usado no pre-bake
C_CURSOR_U8 = np.array([255,   205,   50  ], dtype=np.uint8)     # usado por frame

# ── Fontes ───────────────────────────────────────────────────────────────────

_FONT_FILE = Path(__file__).parent / "fonts" / "epkaisho.ttf"

def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Prioriza fonts/epkaisho.ttf; fallback para fontes do sistema."""
    if _FONT_FILE.exists():
        try:
            return ImageFont.truetype(str(_FONT_FILE), size)
        except Exception:
            pass
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf"  if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── Metadados do áudio ───────────────────────────────────────────────────────

def read_tags(audio_path: str) -> dict:
    """Lê title/artist/album das tags do arquivo de áudio."""
    stem = Path(audio_path).stem
    info = {"title": stem, "artist": "", "album": ""}
    if MutagenFile is None:
        return info
    try:
        m = MutagenFile(audio_path, easy=True)
        if m:
            info["title"]  = str(m.get("title",  [stem])[0])
            info["artist"] = str(m.get("artist", [""])[0])
            info["album"]  = str(m.get("album",  [""])[0])
    except Exception:
        pass
    return info


# ── Fundo borrado ────────────────────────────────────────────────────────────

def make_background(img: Image.Image, W: int = VIDEO_W, H: int = VIDEO_H,
                    is_custom: bool = False, vertical: bool = False) -> np.ndarray:
    """
    Redimensiona e escurece a imagem para usar como fundo.
    Se is_custom=True (fundo enviado pelo usuário), aplica processamento mais leve.
    Retorna array RGB uint8 (H x W x 3).
    """
    bg = img.convert("RGB").resize((W, H), Image.LANCZOS)
    if is_custom:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=6))
        bg = ImageEnhance.Brightness(bg).enhance(0.55)
    else:
        bg = bg.filter(ImageFilter.GaussianBlur(radius=48))
        bg = ImageEnhance.Brightness(bg).enhance(0.30)
    arr = np.array(bg, dtype=np.float32)

    if vertical:
        # Vinheta inferior — gradiente escurecendo a área do texto/waveform
        ys   = np.linspace(0.0, 1.0, H, dtype=np.float32)
        mask = np.clip((ys - 0.35) / 0.65, 0.0, 1.0) ** 1.8 * 120.0
        arr  = np.clip(arr - mask[:, np.newaxis, np.newaxis], 0, 255)
    else:
        # Vinheta lateral direita — gradiente escurecendo a área do texto
        xs   = np.linspace(0.0, 1.0, W, dtype=np.float32)
        mask = np.clip((xs - 0.30) / 0.70, 0.0, 1.0) ** 1.8 * 130.0
        arr  = np.clip(arr - mask[np.newaxis, :, np.newaxis], 0, 255)
    return arr.astype(np.uint8)


# ── Capinha de CD ─────────────────────────────────────────────────────────────

def make_cd_case(img: Image.Image, size: int) -> Image.Image:
    """
    Renderiza a arte do álbum como uma "jewel case" de CD (RGBA PIL Image).
    Inclui sombra, spine, borda e reflexo plástico.
    """
    s   = size
    out = Image.new("RGBA", (s + 30, s + 30), (0, 0, 0, 0))

    # Sombra (múltiplas camadas para suavizar)
    for off, alpha in [(16, 40), (11, 70), (7, 110), (4, 145)]:
        sh = Image.new("RGBA", (s, s), (0, 0, 0, alpha))
        out.paste(sh, (15 + off, 15 + off), sh)

    # Arte
    art = img.convert("RGBA").resize((s, s), Image.LANCZOS)
    out.paste(art, (15, 15), art)

    draw = ImageDraw.Draw(out)
    x0, y0 = 15, 15
    x1, y1 = 15 + s - 1, 15 + s - 1

    # Spine (faixa preta à esquerda)
    draw.rectangle([x0, y0, x0 + 13, y1], fill=(6, 6, 6, 235))
    draw.line([x0 + 14, y0, x0 + 14, y1], fill=(40, 40, 40, 255), width=1)

    # Borda externa
    draw.rectangle([x0, y0, x1, y1], outline=(50, 50, 50, 255), width=1)

    # Reflexo plástico (canto superior-esquerdo)
    shine = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    sd    = ImageDraw.Draw(shine)
    sd.rectangle([0, 0, s // 3, s // 4], fill=(255, 255, 255, 18))
    sd.line([0, 0, s,   0], fill=(255, 255, 255, 60), width=2)
    sd.line([0, 0, 0, s],   fill=(255, 255, 255, 35), width=2)
    out.alpha_composite(shine, (15, 15))

    return out


# ── Disco de CD (círculo visível atrás da capinha) ───────────────────────────

def make_cd_disc(img: Image.Image, diameter: int) -> Image.Image:
    """
    Recorte circular da arte (simula o disco físico visível atrás da capa).
    Adiciona brilho/reflexo metálico.
    """
    d   = diameter
    art = img.convert("RGB").resize((d, d), Image.LANCZOS)
    # Escurece levemente — dá aspecto de vinil/plástico más escuro
    art = ImageEnhance.Brightness(art).enhance(0.55).convert("RGBA")

    # Máscara circular
    mask = Image.new("L", (d, d), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, d - 1, d - 1], fill=255)
    art.putalpha(mask)

    # Buraco central (hub do CD)
    hub_r = d // 14
    cx    = d // 2
    hub_mask = Image.new("L", (d, d), 255)
    ImageDraw.Draw(hub_mask).ellipse(
        [cx - hub_r, cx - hub_r, cx + hub_r, cx + hub_r], fill=0
    )
    art.putalpha(Image.fromarray(
        np.minimum(np.array(mask), np.array(hub_mask)), "L"
    ))

    # Reflexo radial (anel prateado externo)
    overlay = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    od      = ImageDraw.Draw(overlay)
    for i in range(8):
        a = int(30 - i * 3)
        if a <= 0:
            break
        r = d // 2 - i
        od.ellipse([cx - r, cx - r, cx + r, cx + r],
                   outline=(220, 220, 220, a), width=1)

    disc = Image.alpha_composite(art, overlay)
    return disc


# ── Waveform ─────────────────────────────────────────────────────────────────

def extract_spectrum(audio_path: str, fps: int,
                     n_bars: int = N_BARS) -> np.ndarray:
    """
    Espectro mel por frame de vídeo.
    Retorna (n_spec_frames, n_bars) float32 normalizado [0..1].
    Cada linha corresponde a um frame de vídeo — permite waveform reactiva.
    """
    print("  ↳ Analisando espectro…")
    y, sr  = librosa.load(audio_path, mono=True)
    hop    = max(1, sr // fps)
    mel    = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_bars, hop_length=hop, n_fft=2048, power=1.5
    )
    mel_db = librosa.amplitude_to_db(mel, ref=np.max)          # (n_bars, T)
    norm   = np.clip((mel_db + 60.0) / 60.0, 0.0, 1.0)        # 0..1
    spec   = norm.T.astype(np.float32)                         # (T, n_bars)
    return uniform_filter1d(spec, size=3, axis=0)               # suaviza no tempo



# ── Fast numpy compositing ───────────────────────────────────────────────────

def _blend(base_f32: np.ndarray, overlay_u8: np.ndarray,
           ox: int, oy: int) -> None:
    """
    Alpha-composite RGBA uint8 overlay onto RGB float32 base — in-place.
    Handles out-of-bounds coordinates gracefully.
    """
    oh, ow = overlay_u8.shape[:2]
    bh, bw = base_f32.shape[:2]
    sx0 = max(0, -ox);    sy0 = max(0, -oy)
    sx1 = min(ow, bw - ox); sy1 = min(oh, bh - oy)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    dx0, dy0 = ox + sx0, oy + sy0
    dx1, dy1 = ox + sx1, oy + sy1
    src  = overlay_u8[sy0:sy1, sx0:sx1]
    a    = src[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
    rgb  = src[:, :, :3].astype(np.float32)
    dst  = base_f32[dy0:dy1, dx0:dx1]
    dst[:] = rgb * a + dst * (1.0 - a)


def _blend_mask(base: np.ndarray, overlay: np.ndarray,
               ox: int, oy: int) -> None:
    """
    Blend ultra-rápido para overlays de alpha binário (barras da waveform).
    Usa máscara booleana — sem float32, sem multiplicação.
    """
    oh, ow = overlay.shape[:2]
    bh, bw = base.shape[:2]
    sx0 = max(0, -ox);      sy0 = max(0, -oy)
    sx1 = min(ow, bw - ox); sy1 = min(oh, bh - oy)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    dx0, dy0 = ox + sx0, oy + sy0
    dx1, dy1 = ox + sx1, oy + sy1
    src  = overlay[sy0:sy1, sx0:sx1]
    mask = src[:, :, 3] > 127
    base[dy0:dy1, dx0:dx1][mask] = src[mask, :3]


# ── Disco de CD — frames de rotação ─────────────────────────────────────────

def make_disc_rotation_frames(disc_pil: Image.Image,
                               fps: int, rpm: float) -> list:
    """
    Pré-renderiza frames suficientes para um giro completo do disco.
    Retorna lista de arrays RGBA uint8.  RPM=0 → disco parado (1 frame).
    """
    if rpm <= 0:
        return [np.array(disc_pil)]
    n = max(1, int(round(fps * 60.0 / rpm)))
    print(f"  ↳ CD spinning: {n} frames de rotação ({rpm} RPM) pré-renderizados…")
    frames = []
    for i in range(n):
        angle = -(360.0 * i / n)   # sentido horário
        rot = disc_pil.rotate(angle, resample=Image.BILINEAR, expand=False)
        frames.append(np.array(rot))
    return frames


# ── Camada de texto (transparente, composta por último) ──────────────────────

def make_text_layer(info: dict, handle: Optional[str],
                    tx: int, ty: int, W: int, H: int,
                    vertical: bool = False) -> np.ndarray:
    """
    Cria uma camada RGBA transparente com apenas o texto desenhado.
    Composta POR ÚLTIMO no pre-bake para o texto sempre ficar visível
    sobre o disco e a capinha (corrige bug "nome embaixo do CD").
    """
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(layer)

    if vertical:
        f_title  = _load_font(62, bold=True)
        f_artist = _load_font(40)
        f_album  = _load_font(32)
        f_handle = _load_font(26)

        lines = textwrap.wrap(info["title"], width=18)[:2]
        for li, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=f_title)
            tw   = bbox[2] - bbox[0]
            draw.text(((W - tw) // 2, ty + li * 78), line, font=f_title, fill=C_TITLE)

        y_after = ty + len(lines) * 78 + 22

        if info["artist"]:
            bbox = draw.textbbox((0, 0), info["artist"], font=f_artist)
            tw   = bbox[2] - bbox[0]
            draw.text(((W - tw) // 2, y_after), info["artist"], font=f_artist, fill=C_ARTIST)
        if info["album"]:
            bbox = draw.textbbox((0, 0), info["album"], font=f_album)
            tw   = bbox[2] - bbox[0]
            draw.text(((W - tw) // 2, y_after + 56), info["album"], font=f_album, fill=C_ALBUM)

        if handle:
            htxt = f"/ {handle}"
            bbox = draw.textbbox((0, 0), htxt, font=f_handle)
            tw   = bbox[2] - bbox[0]
            draw.text(((W - tw) // 2, H - 72), htxt, font=f_handle, fill=C_HANDLE)
    else:
        f_title  = _load_font(72, bold=True)
        f_artist = _load_font(44)
        f_album  = _load_font(36)
        f_handle = _load_font(28)

        lines = textwrap.wrap(info["title"], width=20)[:2]
        for li, line in enumerate(lines):
            draw.text((tx, ty + li * 88), line, font=f_title, fill=C_TITLE)

        y_after = ty + len(lines) * 88 + 20

        if info["artist"]:
            draw.text((tx, y_after),       info["artist"], font=f_artist, fill=C_ARTIST)
        if info["album"]:
            draw.text((tx, y_after + 62),  info["album"],  font=f_album,  fill=C_ALBUM)

        if handle:
            draw.text((tx, H - 72), f"/ {handle}", font=f_handle, fill=C_HANDLE)

    return np.array(layer)


# ── Gerador principal ────────────────────────────────────────────────────────


def generate_preview(
    image_path:      str,
    title:           Optional[str] = None,
    artist:          Optional[str] = None,
    album:           Optional[str] = None,
    handle:          Optional[str] = None,
    disc_image_path: Optional[str] = None,
    bg_image_path:   Optional[str] = None,
    vertical:        bool          = False,
) -> bytes:
    """
    Gera um único frame PNG do vídeo (sem áudio, sem rotação).
    Retorna os bytes do PNG em memória.
    """
    import io
    if vertical:
        W, H = VIDEO_V_W, VIDEO_V_H
    else:
        W, H = VIDEO_W, VIDEO_H

    img      = Image.open(image_path)
    disc_src = Image.open(disc_image_path) if disc_image_path else img
    bg_src   = Image.open(bg_image_path)   if bg_image_path   else img

    meta = {"title": title or Path(image_path).stem, "artist": artist or "", "album": album or ""}

    # Geometria
    if vertical:
        case_h = int(W * 0.68)
        cd_x   = (W - (case_h + 30)) // 2
        cd_y   = int(H * 0.06)
        disc_d = int(case_h * 0.65)
        disc_x = (W - disc_d) // 2
        disc_y = cd_y + case_h + 30 - disc_d // 2
        tx, ty = 0, disc_y + disc_d + 20
    else:
        case_h = int(H * 0.76)
        cd_x   = int(W * 0.035)
        cd_y   = (H - case_h) // 2 - 8
        disc_d = int(case_h * 0.82)
        disc_x = cd_x + case_h - disc_d // 2 + 20
        disc_y = (H - disc_d) // 2 + 10
        tx, ty = int(W * 0.52), int(H * 0.21)

    bg_arr   = make_background(bg_src, W, H, is_custom=bool(bg_image_path), vertical=vertical)
    disc_pil = make_cd_disc(disc_src, disc_d)
    case_arr = np.array(make_cd_case(img, case_h))
    text_arr = make_text_layer(meta, handle, tx, ty, W, H, vertical=vertical)

    b = bg_arr.astype(np.float32)
    _blend(b, np.array(disc_pil), disc_x, disc_y)
    _blend(b, case_arr,           cd_x,   cd_y)
    _blend(b, text_arr,           0,      0)

    out = Image.fromarray(np.clip(b, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def generate(
    image_path:      str,
    audio_path:      str,
    output_path:     str,
    title:           Optional[str]   = None,
    artist:          Optional[str]   = None,
    album:           Optional[str]   = None,
    handle:          Optional[str]   = None,
    disc_image_path: Optional[str]   = None,   # arte do disco   (None = usa image_path)
    bg_image_path:   Optional[str]   = None,   # imagem de fundo (None = usa image_path)
    fps:             int             = DEF_FPS,
    rpm:             float           = DEF_RPM,
    preset:          str             = DEF_PRESET,
    workers:         int             = DEF_WORKERS,
    progress_cb:     Optional[Callable[[int, str], None]] = None,
    vertical:        bool            = False,
) -> None:
    print(f"\n▶  {Path(audio_path).name}" + (" [VERTICAL]" if vertical else ""))
    if vertical:
        W, H = VIDEO_V_W, VIDEO_V_H
    else:
        W, H = VIDEO_W, VIDEO_H
    _w     = workers if workers > 0 else min(os.cpu_count() or 4, 8)

    # ── Assets ──────────────────────────────────────────────────────────────
    img      = Image.open(image_path)                               # capa (jewel case)
    disc_src = Image.open(disc_image_path) if disc_image_path else img  # arte do disco
    bg_src   = Image.open(bg_image_path)   if bg_image_path   else img  # fundo
    meta = read_tags(audio_path)
    if title:  meta["title"]  = title
    if artist: meta["artist"] = artist
    if album:  meta["album"]  = album

    if progress_cb: progress_cb(5, "Carregando assets…")

    # Duração sem moviepy — usa librosa (rápido, não decodifica tudo)
    try:
        dur = librosa.get_duration(path=audio_path)
    except Exception:
        if MutagenFile:
            m = MutagenFile(audio_path)
            dur = m.info.length if m else 0.0
        else:
            dur = 0.0
    if dur <= 0:
        raise ValueError(f"Não foi possível determinar a duração de: {audio_path}")

    amps = extract_spectrum(audio_path, fps)
    if progress_cb: progress_cb(20, "Espectro analisado…")

    # ── Geometria ────────────────────────────────────────────────────────────
    if vertical:
        case_h = int(W * 0.68)
        cd_x   = (W - (case_h + 30)) // 2
        cd_y   = int(H * 0.06)
        disc_d = int(case_h * 0.65)
        disc_x = (W - disc_d) // 2
        disc_y = cd_y + case_h + 30 - disc_d // 2
        tx, ty = 0, disc_y + disc_d + 20
        wf_x   = int(W * 0.075)
        wf_y   = int(H * 0.82)
        wf_w   = int(W * 0.85)
        wf_h   = int(H * 0.055)
    else:
        case_h = int(H * 0.76)
        cd_x   = int(W * 0.035)
        cd_y   = (H - case_h) // 2 - 8
        disc_d = int(case_h * 0.82)
        disc_x = cd_x + case_h - disc_d // 2 + 20
        disc_y = (H - disc_d) // 2 + 10
        wf_x   = int(W * 0.52)
        wf_y   = int(H * 0.615)
        wf_w   = int(W * 0.44)
        wf_h   = int(H * 0.135)
        tx, ty = int(W * 0.52), int(H * 0.21)

    # ── Pré-renderização (tudo em float32 para blending rápido) ──────────────
    print("  ↳ Preparando visuais…")

    bg_img   = Image.fromarray(make_background(bg_src, W, H, is_custom=bool(bg_image_path), vertical=vertical)).convert("RGBA")
    disc_pil = make_cd_disc(disc_src, disc_d)
    case_arr = np.array(make_cd_case(img, case_h))                  # capa (jewel case)

    # Fundo base como float32 (sem texto — texto será composto por último)
    bg_f     = np.array(bg_img.convert("RGB"), dtype=np.float32)
    # Camada de texto transparente (RGBA) — sempre composta sobre disco e case
    text_arr = make_text_layer(meta, handle, tx, ty, W, H, vertical=vertical)

    # Grade da waveform (constantes pre-computadas, reutilizadas por render_frame)
    frame_amps = amps                                               # alias claro
    bar_total  = max(1, wf_w // N_BARS)
    _xs        = np.arange(wf_w)
    _bar_idx   = np.minimum(_xs // bar_total, N_BARS - 1).astype(np.int32)  # (wf_w,)
    _is_bar    = (_xs % bar_total) < max(1, bar_total - 1)                  # (wf_w,)
    _ys        = np.arange(wf_h, dtype=np.int32)[:, np.newaxis]             # (wf_h, 1)

    if progress_cb: progress_cb(32, "Visuais prontos…")

    # ── Frames de rotação do disco ────────────────────────────────────────────
    disc_frames    = make_disc_rotation_frames(disc_pil, fps, rpm)
    frames_per_rev = len(disc_frames)

    # ── PRE-BAKE: bg + texto + disco[i] + case + waveform_unplayed → uint8 ───
    # Cada frame de rotação vira uma imagem base completa.
    # Custo: frames_per_rev × (W×H×3) bytes  ≈  55 × 6 MB ≈ 330 MB
    # Benefício: por frame de vídeo = 1 memcpy uint8 + 1 blend pequeno.
    mem_mb = frames_per_rev * W * H * 3 / (1024 ** 2)
    print(f"  ↳ Pre-bake: {frames_per_rev} bases × {W}×{H} → {mem_mb:.0f} MB RAM…")
    if progress_cb: progress_cb(46, f"Pre-baking {frames_per_rev} frames…")

    prebaked = []
    for i, disc_f in enumerate(disc_frames):
        b = bg_f.copy()
        _blend(b, disc_f,   disc_x, disc_y)
        _blend(b, case_arr, cd_x,   cd_y)
        _blend(b, text_arr, 0,      0)      # texto SEMPRE por cima do disco e case
        prebaked.append(np.clip(b, 0, 255).astype(np.uint8))
        if progress_cb and i % max(1, frames_per_rev // 10) == 0:
            progress_cb(46 + int(i / frames_per_rev * 6), f"Pre-bake {i+1}/{frames_per_rev}…")

    # libera float32
    del bg_f, disc_frames

    if progress_cb: progress_cb(52, "Pre-bake concluído…")

    # ── Função de render por frame (apenas cópia uint8 + 1 blend) ────────────
    total_frames = int(dur * fps) + 1

    def render_frame(fi: int) -> np.ndarray:
        cur_px = int(min(fi / max(total_frames - 1, 1), 1.0) * wf_w)

        # 1) cópia uint8 da base pré-asada
        frame = prebaked[fi % frames_per_rev].copy()

        # 2) waveform dinâmica — espectro mel do instante atual
        spec_fi   = min(fi, len(frame_amps) - 1)
        amps_f    = frame_amps[spec_fi]                                  # (N_BARS,)
        bhs       = np.maximum(3, (amps_f * wf_h * 0.90).astype(int))  # (N_BARS,)
        by_starts = ((wf_h - bhs) // 2).astype(int)                    # (N_BARS,)
        bhs_c     = bhs[_bar_idx]                                       # (wf_w,)
        by_c      = by_starts[_bar_idx]                                 # (wf_w,)
        bar_mask  = (_ys >= by_c) & (_ys < by_c + bhs_c) & _is_bar     # (wf_h, wf_w)
        cur_bar   = cur_px // max(1, bar_total)
        played_m  = ((_bar_idx < cur_bar) & _is_bar)[np.newaxis, :]    # (1, wf_w)
        wf_region = frame[wf_y : wf_y + wf_h, wf_x : wf_x + wf_w]
        wf_region[bar_mask & ~played_m] = (72,  72,  72)
        wf_region[bar_mask &  played_m] = (255, 255, 255)

        # 3) cursor amarelo (2 px)
        cxp = wf_x + cur_px
        frame[wf_y - 6 : wf_y + wf_h + 7,
              max(0, cxp) : min(W, cxp + 2)] = C_CURSOR_U8

        return frame

    # ── Pipe direto para o FFmpeg ─────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ffcmd = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{W}x{H}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "pipe:0",
        "-i", audio_path,
        "-vcodec", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-preset", preset,
        "-threads", "0",            # usa todos os núcleos no encode x264
        "-acodec", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-movflags", "+faststart",   # streaming-ready (cabeçalho no início)
        "-shortest", output_path,
    ]

    print(f"  ↳ {total_frames} frames @ {fps}fps · {_w} workers · preset={preset} · rpm={rpm}  →  {output_path}")

    BATCH  = max(fps, _w * 4)
    proc   = subprocess.Popen(ffcmd, stdin=subprocess.PIPE, bufsize=10 ** 7,
                               stderr=subprocess.DEVNULL)
    try:
        with ThreadPoolExecutor(max_workers=_w) as pool:
            for bs in range(0, total_frames, BATCH):
                be     = min(bs + BATCH, total_frames)
                frames = list(pool.map(render_frame, range(bs, be)))
                for f in frames:
                    proc.stdin.write(f.tobytes())
                if progress_cb:
                    pct = 50 + int(be / total_frames * 45)
                    progress_cb(min(pct, 95),
                                f"Renderizando {be}/{total_frames} frames…")
    except Exception:
        proc.kill()
        raise
    finally:
        proc.stdin.close()
        rc = proc.wait()

    if rc != 0:
        raise RuntimeError(f"FFmpeg encerrou com código {rc}")

    if progress_cb: progress_cb(99, "Finalizando…")
    print(f"  ✓  Salvo em: {output_path}\n")


# ── Modo batch ────────────────────────────────────────────────────────────────

def batch_generate(folder: str, handle: Optional[str] = None,
                   fps: int = DEF_FPS, rpm: float = DEF_RPM,
                   preset: str = DEF_PRESET, workers: int = DEF_WORKERS,
                   vertical: bool = False) -> None:
    """
    Processa toda uma pasta.
    Para cada áudio, procura uma imagem de mesmo nome-base; se não encontrar,
    usa a primeira imagem disponível na pasta como fallback.
    """
    folder_p = Path(folder)
    AUDIO_EX = {".mp3", ".flac", ".wav", ".ogg", ".m4a", ".aac"}
    IMAGE_EX = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    audios   = sorted(f for f in folder_p.iterdir()
                      if f.suffix.lower() in AUDIO_EX)
    if not audios:
        sys.exit(f"Nenhum áudio encontrado em: {folder_p}")

    fallback_img = next(
        (f for f in sorted(folder_p.iterdir()) if f.suffix.lower() in IMAGE_EX),
        None
    )

    for aud in audios:
        img = next(
            (folder_p / (aud.stem + ext) for ext in IMAGE_EX
             if (folder_p / (aud.stem + ext)).exists()),
            fallback_img,
        )
        if img is None:
            print(f"  Pulando {aud.name} — nenhuma imagem encontrada")
            continue

        out = str(folder_p / "output" / f"{aud.stem}.mp4")
        generate(str(img), str(aud), out, handle=handle,
                 fps=fps, rpm=rpm, preset=preset, workers=workers,
                 vertical=vertical)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="cdmaker",
        description="CDMaker — Gerador de vídeos Now Playing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Exemplos:
              python main.py capa.jpg musica.mp3 --handle NESTALGIA
              python main.py capa.jpg musica.mp3 --title "La Colina Esmeralda" --artist "Sonic" --album "NESTALGIA"
              python main.py --batch ./musicas --handle NESTALGIA
            """
        ),
    )

    ap.add_argument("image", nargs="?", help="Caminho da imagem de capa")
    ap.add_argument("audio", nargs="?", help="Caminho do arquivo de áudio")
    ap.add_argument("-o", "--output", default=None,
                    help="Caminho do vídeo de saída (.mp4)")
    ap.add_argument("--title",  default=None, help="Título da música (sobrescreve tags)")
    ap.add_argument("--artist", default=None, help="Nome do artista")
    ap.add_argument("--album",  default=None, help="Nome do álbum / gravadora")
    ap.add_argument("--handle", default=None,
                    help="Handle das redes sociais (ex: NESTALGIA)")
    ap.add_argument("--batch", metavar="PASTA", default=None,
                    help="Processa todos os áudios de uma pasta")
    ap.add_argument("--rpm",     type=float, default=DEF_RPM,
                    help=f"Velocidade de rotação do CD em RPM (0=parado, padrão={DEF_RPM})")
    ap.add_argument("--fps",     type=int,   default=DEF_FPS,
                    help=f"Frames por segundo do vídeo (padrão={DEF_FPS})")
    ap.add_argument("--preset",  default=DEF_PRESET,
                    choices=["ultrafast","superfast","veryfast","faster","fast","medium","slow"],
                    help=f"Preset FFmpeg x264 (padrão={DEF_PRESET})")
    ap.add_argument("--workers", type=int, default=DEF_WORKERS,
                    help="Threads para gerar frames (0=auto)")
    ap.add_argument("--vertical", action="store_true",
                    help="Gera vídeo vertical 1080×1920 (TikTok / Shorts)")

    args = ap.parse_args()

    # ── Modo batch ────────────────────────────────────────────────────────────
    if args.batch:
        batch_generate(args.batch, handle=args.handle,
                       fps=args.fps, rpm=args.rpm,
                       preset=args.preset, workers=args.workers,
                       vertical=args.vertical)
        return

    # ── Modo individual ───────────────────────────────────────────────────────
    if not (args.image and args.audio):
        ap.print_help()
        sys.exit(1)

    for filepath, label in [(args.image, "Imagem"), (args.audio, "Áudio")]:
        if not os.path.exists(filepath):
            sys.exit(f"Erro: {label} não encontrado: {filepath}")

    output = args.output or str(
        Path("output") / (Path(args.audio).stem + ".mp4")
    )

    generate(
        args.image,
        args.audio,
        output,
        title=args.title,
        artist=args.artist,
        album=args.album,
        handle=args.handle,
        fps=args.fps,
        rpm=args.rpm,
        preset=args.preset,
        workers=args.workers,
        vertical=args.vertical,
    )


if __name__ == "__main__":
    main()
