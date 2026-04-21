"""
FootScan Backend v4 — Flask API (Simplified)
Pengukur Ukuran Kaki — Deteksi Andal & Cepat

Penyederhanaan dari v3:
  - Hapus GrabCut (lambat, sering gagal di foto nyata)
  - 3 metode segmentasi yang lebih fokus: Skin, Edge, Threshold
  - Scoring kontur disederhanakan tapi tetap akurat
  - Kode lebih ringkas, mudah dipelihara

Requirements:
    pip install flask flask-cors opencv-python-headless numpy Pillow

Run:
    python app.py  →  http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image, ImageOps
import base64, io, os, logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# ─────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────
REFERENCE_SIZES = {
    "a4":          {"width": 21.0,  "height": 29.7},
    "credit_card": {"width": 8.56,  "height": 5.40},
    "ruler_30":    {"width": 30.0,  "height": 3.0},
    "none":        None,
}

SHOE_TABLE = [
    (20.0, 20.6, "32","32","2","1"),    (20.6, 21.3, "33","33","3","2"),
    (21.3, 21.9, "34","34","3.5","2.5"),(21.9, 22.5, "35","35","4","3"),
    (22.5, 23.2, "36","36","5","4"),    (23.2, 23.8, "37","37","6","5"),
    (23.8, 24.5, "38","38","6.5","5.5"),(24.5, 25.1, "39","39","7","6"),
    (25.1, 25.8, "40","40","8","7"),    (25.8, 26.4, "41","41","8.5","7.5"),
    (26.4, 27.0, "42","42","9","8"),    (27.0, 27.7, "43","43","10","9"),
    (27.7, 28.3, "44","44","10.5","9.5"),(28.3, 29.0, "45","45","11","10"),
    (29.0, 29.6, "46","46","12","11"),  (29.6, 30.3, "47","47","13","12"),
]


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def pil_to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def cv_to_b64(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Encode gagal")
    return base64.b64encode(buf.tobytes()).decode()

def resize_for_processing(img, max_dim=1200):
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img, scale

def get_shoe_sizes(length_cm):
    eff = length_cm + 1.5
    for mn, mx, id_, eu, us, uk in SHOE_TABLE:
        if mn <= eff < mx:
            return {"ID": id_, "EU": eu, "US": us, "UK": uk}
    if eff >= 30.3:
        return {"ID": ">47", "EU": ">47", "US": ">13", "UK": ">12"}
    return {"ID": "<32", "EU": "<32", "US": "<2", "UK": "<1"}


# ═══════════════════════════════════════════════════════
# METODE SEGMENTASI (3 metode sederhana & andal)
# ═══════════════════════════════════════════════════════

def mask_skin(img_bgr):
    """
    Deteksi warna kulit via YCrCb + HSV.
    Metode paling cepat, andal untuk kulit terang-gelap.
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    m_ycrcb = cv2.inRange(ycrcb,
        np.array([0, 125,  70], np.uint8),
        np.array([255, 185, 135], np.uint8))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m_hsv_a = cv2.inRange(hsv, np.array([0,  15, 40], np.uint8), np.array([22, 200, 255], np.uint8))
    m_hsv_b = cv2.inRange(hsv, np.array([330,15, 40], np.uint8), np.array([180,200, 255], np.uint8))
    m_hsv   = cv2.bitwise_or(m_hsv_a, m_hsv_b)

    skin = cv2.bitwise_and(m_ycrcb, m_hsv)

    # Tutup lubang kecil, hilangkan noise
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, k_close)
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN,  k_open)
    return skin


def mask_edge(img_bgr):
    """
    Deteksi berdasarkan tepi Canny + flood-fill dari sudut.
    Bekerja baik saat warna kaki mirip background tapi bentuk kaki jelas.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.5, (8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Dua skala Canny, gabungkan
    edges = cv2.bitwise_or(
        cv2.Canny(blur, 20, 60),
        cv2.Canny(blur, 40, 120)
    )
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

    # Flood-fill dari pojok → area yang tidak terjangkau = foreground
    filled = cv2.bitwise_not(edges)
    ff_mask = np.zeros((h+2, w+2), np.uint8)
    for sx, sy in [(0,0),(0,h-1),(w-1,0),(w-1,h-1),(w//2,0),(w//2,h-1),(0,h//2),(w-1,h//2)]:
        if filled[sy, sx] == 255:
            cv2.floodFill(filled, ff_mask, (sx, sy), 128)

    result = np.where(filled == 255, 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (12, 12))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)))
    return result


def mask_threshold(img_bgr, threshold_val):
    """
    Threshold adaptif + Otsu sebagai fallback terakhir.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(3.0, (6, 6)).apply(gray)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    _, m_otsu   = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, m_manual = cv2.threshold(blur, threshold_val, 255, cv2.THRESH_BINARY_INV)
    combined = cv2.bitwise_or(m_otsu, m_manual)

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k_close)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)
    return combined


# ═══════════════════════════════════════════════════════
# SCORING & PEMILIHAN KONTUR
# ═══════════════════════════════════════════════════════

def score_contour(contour, img_h, img_w):
    """
    Skor 0–100: seberapa besar kemungkinan kontur ini adalah kaki manusia.
    Kriteria: aspect ratio, ukuran relatif, solidity, dan posisi tepi.
    """
    area = cv2.contourArea(contour)
    if area < 300:
        return 0.0

    img_area = img_h * img_w

    # 1. Aspect ratio — kaki umumnya 1.8–3.8 (panjang ÷ lebar)
    _, (rw, rh), _ = cv2.minAreaRect(contour)
    if rw == 0 or rh == 0:
        return 0.0
    aspect = max(rw, rh) / min(rw, rh)
    if   1.8 <= aspect <= 3.8:  asp_s = 100
    elif 1.4 <= aspect <= 5.0:  asp_s = 50
    else:                        asp_s = 0

    # 2. Area relatif — kaki umumnya 4%–50% frame
    ar = area / img_area
    if   0.04 <= ar <= 0.50: area_s = 100
    elif 0.02 <= ar <= 0.70: area_s = 40
    else:                     area_s = 0

    # 3. Solidity — kaki cukup solid (0.60–0.96)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity  = area / hull_area if hull_area > 0 else 0
    if   0.60 <= solidity <= 0.96: sol_s = 100
    elif 0.45 <= solidity:         sol_s = 45
    else:                           sol_s = 0

    # 4. Edge penalty — kontur yang menyentuh 2+ sisi frame kemungkinan background
    x, y, cw, ch = cv2.boundingRect(contour)
    margin = max(3, int(min(img_h, img_w) * 0.01))
    touches = int(x <= margin) + int(y <= margin) + \
              int(x + cw >= img_w - margin) + int(y + ch >= img_h - margin)
    edge_s = 100 if touches <= 1 else (50 if touches == 2 else 0)

    return asp_s * 0.35 + area_s * 0.25 + sol_s * 0.25 + edge_s * 0.15


def pick_best_contour(mask, img_h, img_w):
    """Temukan kontur terbaik dari mask berdasarkan scoring."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    candidates = [(score_contour(c, img_h, img_w), c) for c in contours if cv2.contourArea(c) > 200]
    if not candidates:
        return None, 0

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_c = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 0
    confidence = min(97, int(best_score * 0.65 + (best_score - second_score) * 0.35))
    return best_c, confidence


# ═══════════════════════════════════════════════════════
# PIPELINE SEGMENTASI UTAMA
# ═══════════════════════════════════════════════════════

def segment_foot(img_bgr, threshold_val=127):
    """
    Coba 3 metode segmentasi secara cascade.
    Pilih hasil dengan skor kontur tertinggi.
    """
    h, w = img_bgr.shape[:2]
    results = []  # (method_name, contour, confidence, score)

    methods = [
        ("skin_color", lambda: mask_skin(img_bgr)),
        ("edge_based",  lambda: mask_edge(img_bgr)),
        ("threshold",   lambda: mask_threshold(img_bgr, threshold_val)),
    ]

    best_mask = None
    for name, fn in methods:
        try:
            mask = fn()
            # Minimal coverage check
            if np.count_nonzero(mask) / (h * w) < 0.005:
                logger.info(f"{name}: mask terlalu kecil, skip")
                continue
            c, conf = pick_best_contour(mask, h, w)
            if c is None:
                continue
            sc = score_contour(c, h, w)
            logger.info(f"{name}: score={sc:.1f} conf={conf}")
            results.append((name, mask, c, conf, sc))
        except Exception as e:
            logger.warning(f"{name} gagal: {e}")

    if not results:
        raise ValueError(
            "Semua metode segmentasi gagal. "
            "Tips: pastikan kaki kontras dengan latar, foto dari atas, cahaya cukup."
        )

    results.sort(key=lambda x: x[4], reverse=True)
    best_name, best_mask, best_c, best_conf, best_sc = results[0]
    logger.info(f"Dipilih: {best_name} (score={best_sc:.1f})")

    return best_mask, best_name, best_c, best_conf


# ═══════════════════════════════════════════════════════
# VALIDASI KONTUR
# ═══════════════════════════════════════════════════════

def validate_contour(contour, img_h, img_w):
    """Pastikan kontur benar-benar menyerupai kaki."""
    if contour is None:
        raise ValueError(
            "Kontur kaki tidak terdeteksi. "
            "Tips: foto dari atas (tegak lurus), gunakan alas berwarna kontras."
        )
    sc = score_contour(contour, img_h, img_w)
    if sc < 15:
        raise ValueError(
            f"Kontur terdeteksi tapi tidak menyerupai kaki (skor={sc:.0f}/100). "
            "Coba kurangi threshold atau ubah posisi kaki agar lebih vertikal."
        )
    return sc


# ═══════════════════════════════════════════════════════
# DIMENSI & LUAS
# ═══════════════════════════════════════════════════════

def compute_dimensions(contour):
    _, (rw, rh), angle = cv2.minAreaRect(contour)
    x, y, bw, bh = cv2.boundingRect(contour)
    return {
        "length_px": float(max(rw, rh)),
        "width_px":  float(min(rw, rh)),
        "bbox_x": x, "bbox_y": y, "bbox_w": bw, "bbox_h": bh,
        "angle": float(angle),
    }


def compute_area(contour):
    """
    Luas dengan Shoelace Formula (exact untuk polygon) diverifikasi cv2.
    Rata-rata keduanya untuk stabilitas.
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    shoelace = 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    cv2_area = cv2.contourArea(contour)
    final = (shoelace + cv2_area) / 2.0
    logger.info(f"Area — shoelace:{shoelace:.0f} cv2:{cv2_area:.0f} final:{final:.0f} px²")
    return final


# ═══════════════════════════════════════════════════════
# KALIBRASI
# ═══════════════════════════════════════════════════════

def compute_pixel_scale(img_bgr, ref_key):
    """Deteksi objek referensi persegi panjang untuk skala piksel/cm."""
    if ref_key == "none" or ref_key not in REFERENCE_SIZES:
        return None
    ref = REFERENCE_SIZES[ref_key]
    if ref is None:
        return None

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref_long  = max(ref["width"], ref["height"])
    ref_short = min(ref["width"], ref["height"])
    best_scale, best_score = None, 0

    for c in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
        if cv2.contourArea(c) < 500:
            continue
        approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue
        _, (cw, ch), _ = cv2.minAreaRect(c)
        dim_long, dim_short = max(cw, ch), min(cw, ch)
        if dim_long == 0 or dim_short == 0:
            continue
        # Aspect ratio harus cocok ±25%
        if abs(ref_long/ref_short - dim_long/dim_short) / (ref_long/ref_short) > 0.25:
            continue
        sl, ss = dim_long/ref_long, dim_short/ref_short
        if abs(sl - ss) / max(sl, ss) > 0.20:
            continue
        area = cv2.contourArea(c)
        if area > best_score:
            best_score = area
            best_scale = (sl + ss) / 2.0

    return best_scale


# ═══════════════════════════════════════════════════════
# VISUALISASI
# ═══════════════════════════════════════════════════════

def generate_visualization(img_bgr, contour, dims, length_cm, width_cm, area_cm2, method):
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # Overlay kontur semi-transparan
    overlay = vis.copy()
    cv2.drawContours(overlay, [contour], -1, (74, 240, 176), -1)
    cv2.addWeighted(overlay, 0.28, vis, 0.72, 0, vis)
    cv2.drawContours(vis, [contour], -1, (74, 240, 176), 3)

    # Bounding box
    bx, by, bw, bh = dims["bbox_x"], dims["bbox_y"], dims["bbox_w"], dims["bbox_h"]
    cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (56, 178, 255), 2)

    # Garis dimensi
    mid_x, mid_y = bx + bw//2, by + bh//2
    cv2.arrowedLine(vis, (mid_x, by),     (mid_x, by+bh), (255, 200, 80), 2, tipLength=0.02)
    cv2.arrowedLine(vis, (mid_x, by+bh),  (mid_x, by),    (255, 200, 80), 2, tipLength=0.02)
    cv2.arrowedLine(vis, (bx, mid_y),     (bx+bw, mid_y), (255, 130, 180), 2, tipLength=0.02)
    cv2.arrowedLine(vis, (bx+bw, mid_y),  (bx, mid_y),    (255, 130, 180), 2, tipLength=0.02)

    def label(img, text, pos, color, bg=(15, 20, 30)):
        fs = max(0.45, min(w, h) / 1000)
        th = 1 if fs < 0.7 else 2
        (tw, txh), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x, y = pos
        cv2.rectangle(img, (x-4, y-txh-4), (x+tw+4, y+bl+2), bg, -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)

    label(vis, f"P: {length_cm:.1f}cm", (mid_x+8, by+bh//2-10), (255, 200, 80))
    label(vis, f"L: {width_cm:.1f}cm",  (bx+8, mid_y-14),        (255, 130, 180))
    label(vis, f"A: {area_cm2:.1f}cm2", (bx+4, by+bh-8),          (74, 240, 176))

    # Title bar atas
    bar_h = 52
    bar = np.full((bar_h, w, 3), (15, 20, 30), np.uint8)
    vis = np.vstack([bar, vis])
    fs = max(0.4, min(w, h) / 1200)
    cv2.putText(vis, f"FootScan | P={length_cm:.1f}cm W={width_cm:.1f}cm A={area_cm2:.1f}cm2 [{method}]",
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, fs, (74, 240, 176), 1, cv2.LINE_AA)
    return vis


# ═══════════════════════════════════════════════════════
# FUNGSI ANALISIS UTAMA
# ═══════════════════════════════════════════════════════

def analyze_foot(image_bytes, ref_key="a4", threshold=127):
    # 1. Load & auto-rotate EXIF
    pil_img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    img_orig = pil_to_cv(pil_img)

    # 2. Resize untuk kecepatan
    img, scale = resize_for_processing(img_orig, max_dim=1200)
    h, w = img.shape[:2]
    logger.info(f"Gambar: {img_orig.shape[1]}x{img_orig.shape[0]} → {w}x{h} (scale={scale:.2f})")

    # 3. Segmentasi
    mask, method, contour, confidence = segment_foot(img, threshold)

    # 4. Validasi
    validate_contour(contour, h, w)

    # 5. Dimensi & luas
    dims   = compute_dimensions(contour)
    area_px2 = compute_area(contour)

    # 6. Kalibrasi skala piksel/cm
    px_per_cm = compute_pixel_scale(img, ref_key)
    calib_src = "reference_object"
    if not px_per_cm or px_per_cm <= 0:
        px_per_cm = dims["length_px"] / 24.0
        calib_src = "estimasi_24cm"
        logger.warning(f"Fallback scale: {px_per_cm:.2f} px/cm")
    else:
        logger.info(f"Referensi terdeteksi: {px_per_cm:.2f} px/cm")

    # 7. Konversi ke cm
    length_cm = round(dims["length_px"] / px_per_cm, 1)
    width_cm  = round(dims["width_px"]  / px_per_cm, 1)
    area_cm2  = round(area_px2          / (px_per_cm**2), 1)

    # 8. Sanity check — rekalibrasi jika di luar jangkauan fisik kaki
    if not (17.0 <= length_cm <= 38.0):
        logger.warning(f"Panjang {length_cm}cm di luar range, rekalibrasi ke 24cm")
        px_per_cm = dims["length_px"] / 24.0
        calib_src = "rekalibrasi_24cm"
        length_cm = 24.0
        width_cm  = round(dims["width_px"]  / px_per_cm, 1)
        area_cm2  = round(area_px2          / (px_per_cm**2), 1)

    # 9. Ukuran sepatu
    shoe_sizes = get_shoe_sizes(length_cm)

    # 10. Visualisasi (pada gambar resolusi asli)
    if scale < 1.0:
        contour_orig = (contour.astype(np.float32) / scale).astype(np.int32)
        dims_orig    = compute_dimensions(contour_orig)
    else:
        contour_orig = contour
        dims_orig    = dims

    vis = generate_visualization(img_orig, contour_orig, dims_orig,
                                 length_cm, width_cm, area_cm2, method)
    contour_b64 = cv_to_b64(vis)

    # 11. Export titik kontur dalam cm (untuk chart Kartesius di frontend)
    raw  = contour.reshape(-1, 2).astype(np.float64)
    step = max(1, len(raw) // 300)
    raw  = raw[::step]
    cx_px = (raw[:, 0].max() + raw[:, 0].min()) / 2
    cy_px = (raw[:, 1].max() + raw[:, 1].min()) / 2
    pts_cm = [[round((float(p[0])-cx_px)/px_per_cm, 3),
               round(-(float(p[1])-cy_px)/px_per_cm, 3)] for p in raw]

    # 12. Shoelace steps (sampel 12 titik untuk tabel edukasi)
    pts_arr = np.array(pts_cm)
    n_pts   = len(pts_arr)
    sl_steps = []
    for i in range(min(n_pts, 12)):
        j  = (i+1) % n_pts
        xi, yi = float(pts_arr[i][0]), float(pts_arr[i][1])
        xj, yj = float(pts_arr[j][0]), float(pts_arr[j][1])
        p1 = round(xi*yj, 4); p2 = round(xj*yi, 4)
        sl_steps.append({"i": i, "xi": round(xi,3), "yi": round(yi,3),
                          "xj": round(xj,3), "yj": round(yj,3),
                          "prod1": p1, "prod2": p2, "delta": round(p1-p2, 4)})

    # 13. Trapezoid strips (20 strip untuk tabel edukasi)
    xv, yv = pts_arr[:, 0], pts_arr[:, 1]
    xmn, xmx = float(xv.min()), float(xv.max())
    dx = (xmx - xmn) / 20
    trap_strips = []
    total_trap  = 0.0
    for i in range(20):
        x0 = xmn + i*dx; x1 = x0 + dx
        col = (xv >= x0 - dx*0.6) & (xv <= x1 + dx*0.6)
        yt = float(yv[col].max()) if col.sum() > 0 else 0.0
        yb = float(yv[col].min()) if col.sum() > 0 else 0.0
        h0 = yt - yb; as_ = round(h0 * abs(dx), 4)
        total_trap += as_
        trap_strips.append({"i": i, "x0": round(x0,3), "x1": round(x1,3),
                             "dx": round(abs(dx),3), "y_top": round(yt,3),
                             "y_bot": round(yb,3), "h": round(h0,3), "area": as_})

    logger.info(f"HASIL: P={length_cm}cm W={width_cm}cm A={area_cm2}cm² "
                f"sepatu={shoe_sizes} conf={confidence}%")

    return {
        "length_cm": length_cm, "width_cm": width_cm, "area_cm2": area_cm2,
        "shoe_sizes": shoe_sizes,
        "confidence": confidence, "threshold_used": threshold,
        "px_per_cm": round(px_per_cm, 2), "pixel_scale": round(px_per_cm, 2),
        "image_size": {"width": w, "height": h},
        "contour_image": contour_b64,
        "contour_points": len(contour), "n_contour_pts": len(contour),
        "calibration_source": calib_src,
        "detection_method": method,
        "contour_pts_cm": pts_cm,
        "shoelace_steps": sl_steps,
        "trap_strips": trap_strips,
        "trap_total_cm2": round(total_trap, 2),
        "area_px2": round(float(area_px2), 1),
        "dx_avg": round(float(dims["length_px"]) / len(contour), 3),
        "x_range_px": int(dims["bbox_w"]), "y_range_px": int(dims["bbox_h"]),
    }


# ═══════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "FootScan API v4", "version": "4.0.0"})


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar."}), 400
    f = request.files["image"]
    if f.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400
    if f.content_type not in {"image/jpeg", "image/png", "image/jpg"}:
        return jsonify({"error": f"Format tidak didukung: {f.content_type}"}), 400

    ref_key = request.form.get("reference", "a4")
    try:    threshold = max(30, min(240, int(request.form.get("threshold", 127))))
    except: threshold = 127

    try:
        image_bytes = f.read()
    except Exception as e:
        return jsonify({"error": f"Gagal baca file: {e}"}), 400

    try:
        result = analyze_foot(image_bytes, ref_key=ref_key, threshold=threshold)
        return jsonify(result), 200
    except ValueError as e:
        logger.warning(f"Validasi: {e}")
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        logger.error(f"Analisis gagal: {e}", exc_info=True)
        return jsonify({
            "error": "Analisis gagal. Coba foto ulang dengan latar yang kontras.",
            "detail": str(e)
        }), 500


@app.route("/")
def index():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html"}
    return "FootScan API v4 running.", 200


if __name__ == "__main__":
    print("=" * 60)
    print("  FootScan Backend v4 — Simplified Foot Detection")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5000))
    print(f"  URL    : http://0.0.0.0:{port}")
    print(f"  Health : http://0.0.0.0:{port}/health")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
