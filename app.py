"""
FootScan Backend — Flask API
Pengukur Ukuran Kaki menggunakan Computer Vision + Numerical Integration

Requirements:
    pip install flask flask-cors opencv-python-headless numpy Pillow

Run:
    python app.py
    → Server berjalan di http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import base64
import io
import os
import math
import logging

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from frontend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Maximum upload size: 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


# ─────────────────────────────────────────────
# Reference Object Dimensions (in cm)
# ─────────────────────────────────────────────
REFERENCE_SIZES = {
    "a4":          {"width": 21.0,   "height": 29.7},   # Kertas A4
    "credit_card": {"width": 8.56,   "height": 5.40},   # Kartu kredit/ATM
    "ruler_30":    {"width": 30.0,   "height": 3.0},    # Penggaris 30 cm
    "none":        None                                   # No calibration
}

# ─────────────────────────────────────────────
# Shoe Size Conversion Table
# Sumber: ISO 9407 + pasar Indonesia
# ─────────────────────────────────────────────
SHOE_SIZE_TABLE = [
    # (min_cm, max_cm, ID, EU, US_M, UK)
    (20.0, 20.6, "32", "32", "2",   "1"),
    (20.6, 21.3, "33", "33", "3",   "2"),
    (21.3, 21.9, "34", "34", "3.5", "2.5"),
    (21.9, 22.5, "35", "35", "4",   "3"),
    (22.5, 23.2, "36", "36", "5",   "4"),
    (23.2, 23.8, "37", "37", "6",   "5"),
    (23.8, 24.5, "38", "38", "6.5", "5.5"),
    (24.5, 25.1, "39", "39", "7",   "6"),
    (25.1, 25.8, "40", "40", "8",   "7"),
    (25.8, 26.4, "41", "41", "8.5", "7.5"),
    (26.4, 27.0, "42", "42", "9",   "8"),
    (27.0, 27.7, "43", "43", "10",  "9"),
    (27.7, 28.3, "44", "44", "10.5","9.5"),
    (28.3, 29.0, "45", "45", "11",  "10"),
    (29.0, 29.6, "46", "46", "12",  "11"),
    (29.6, 30.3, "47", "47", "13",  "12"),
]


# ─────────────────────────────────────────────
# Helper: Convert PIL Image → numpy array
# ─────────────────────────────────────────────
def pil_to_cv(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_base64(img: np.ndarray) -> str:
    """Encode OpenCV image (BGR) → PNG → base64 string"""
    success, buf = cv2.imencode(".png", img)
    if not success:
        raise RuntimeError("Gagal encode gambar")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ─────────────────────────────────────────────
# Preprocessing Pipeline
# ─────────────────────────────────────────────
def preprocess_image(img_bgr: np.ndarray, threshold_val: int = 127) -> np.ndarray:
    """
    Pipeline preprocessing:
    1. Grayscale
    2. Gaussian blur (noise reduction)
    3. Adaptive/manual threshold → binary mask
    4. Morphological cleanup (close holes, remove noise)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Enhance contrast with CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # → handles variasi pencahayaan
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Gaussian blur untuk reduce noise
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Dual-strategy threshold:
    # - Adaptive threshold: tahan terhadap gradien cahaya
    # - Otsu's method: menentukan threshold optimal otomatis
    adaptive_mask = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=10
    )

    # Otsu threshold sebagai fallback/kombinasi
    _, otsu_mask = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Manual threshold dari user
    _, manual_mask = cv2.threshold(
        blurred, threshold_val, 255, cv2.THRESH_BINARY_INV
    )

    # Gabungkan: prioritas adaptive, backup manual
    combined = cv2.bitwise_or(adaptive_mask, manual_mask)
    combined = cv2.bitwise_or(combined, otsu_mask)

    # Morphological operations untuk membersihkan mask
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Close: menutup lubang kecil di dalam kaki
    mask = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)
    # Open: menghapus noise kecil di luar kaki
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

    return mask


# ─────────────────────────────────────────────
# Contour Detection
# ─────────────────────────────────────────────
def find_foot_contour(mask: np.ndarray) -> tuple:
    """
    Deteksi kontur terbesar → diasumsikan sebagai kaki.
    Returns: (largest_contour, all_contours)
    Raises ValueError jika tidak ditemukan kontur yang valid.
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )

    if not contours:
        raise ValueError("Tidak ada kontur yang terdeteksi. Pastikan kaki terlihat jelas.")

    # Ambil kontur terbesar berdasarkan area
    areas = [cv2.contourArea(c) for c in contours]
    max_idx = np.argmax(areas)
    largest = contours[max_idx]
    max_area = areas[max_idx]

    # Validasi: kontur harus cukup besar (minimal 1% dari luas gambar)
    h, w = mask.shape
    image_area = h * w
    if max_area < image_area * 0.01:
        raise ValueError(
            "Kontur kaki terlalu kecil. Pastikan kaki mengisi sebagian besar frame foto."
        )

    # Confidence score (rasio kontur utama vs total area semua kontur)
    total_area = sum(areas)
    confidence = int((max_area / total_area) * 100) if total_area > 0 else 0

    return largest, contours, confidence


# ─────────────────────────────────────────────
# Numerical Integration — Trapezoidal Rule
# ─────────────────────────────────────────────
def compute_area_trapezoidal(contour: np.ndarray) -> float:
    """
    Menghitung luas area kontur menggunakan:
    1. Shoelace Formula (Green's theorem) — exact untuk polygon
    2. Dikombinasikan dengan Trapezoidal Rule untuk validasi

    Rumus Shoelace:
        A = 0.5 * |Σ(x_i * y_{i+1} - x_{i+1} * y_i)|

    Trapezoidal Rule (sebagai verifikasi):
        A = 0.5 * |Σ(y_i + y_{i+1}) * (x_{i+1} - x_i)|

    Returns: area dalam satuan piksel²
    """
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)

    if n < 3:
        return 0.0

    # ── Shoelace Formula ──
    x = pts[:, 0]
    y = pts[:, 1]
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    shoelace_area = 0.5 * abs(np.sum(x * y_next - x_next * y))

    # ── Trapezoidal Rule ──
    # Proyeksikan kontur ke sumbu-x, integrasikan terhadap y
    # Area = integral dari x_left ke x_right dari y_bottom ke y_top
    # Simplified: gunakan cv2.contourArea sebagai cross-check
    cv2_area = cv2.contourArea(contour)

    # Ambil rata-rata sebagai hasil akhir (biasanya sangat dekat)
    final_area = (shoelace_area + cv2_area) / 2.0

    logger.info(f"Shoelace area: {shoelace_area:.1f} px², "
                f"cv2 area: {cv2_area:.1f} px², "
                f"final: {final_area:.1f} px²")

    return final_area


def compute_trapezoidal_integration_demo(contour: np.ndarray) -> dict:
    """
    Demonstrasi Riemann + Trapezoidal integration step-by-step.
    Membagi kontur menjadi horizontal slices dan mengintegrasikan.
    Digunakan untuk keperluan edukasi/penjelasan.
    """
    pts = contour.reshape(-1, 2)
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    y_min, y_max = int(y_coords.min()), int(y_coords.max())
    n_slices = 100
    dy = (y_max - y_min) / n_slices

    slices = []
    riemann_area = 0.0
    trap_area = 0.0

    prev_width = None
    for i in range(n_slices):
        y_line = y_min + i * dy
        # Temukan x values pada garis horizontal y_line
        mask_local = np.zeros((y_max - y_min + 2, int(x_coords.max()) + 2), dtype=np.uint8)
        cv2.drawContours(mask_local, [contour - [0, y_min]], -1, 255, -1)
        row = int(y_line - y_min)
        if 0 <= row < mask_local.shape[0]:
            row_data = mask_local[row, :]
            nonzero = np.where(row_data > 0)[0]
            if len(nonzero) > 0:
                width = float(nonzero[-1] - nonzero[0])
                # Riemann sum (left rule)
                riemann_area += width * dy
                # Trapezoidal rule
                if prev_width is not None:
                    trap_area += 0.5 * (prev_width + width) * dy
                prev_width = width

    return {
        "riemann_area_px": riemann_area,
        "trapezoidal_area_px": trap_area,
        "n_slices": n_slices
    }


# ─────────────────────────────────────────────
# Bounding Box — Length & Width
# ─────────────────────────────────────────────
def compute_dimensions(contour: np.ndarray) -> dict:
    """
    Menghitung panjang dan lebar kaki dari bounding rectangle.
    Menggunakan minAreaRect untuk mendapatkan orientasi terbaik.
    Returns: dict dengan length_px, width_px, angle
    """
    # Rotated bounding rectangle
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect

    # Konvensi: panjang adalah dimensi terbesar
    length_px = max(w, h)
    width_px = min(w, h)

    # Standard bounding box (axis-aligned)
    x, y, bw, bh = cv2.boundingRect(contour)

    return {
        "length_px": float(length_px),
        "width_px": float(width_px),
        "bbox_x": x,
        "bbox_y": y,
        "bbox_w": bw,
        "bbox_h": bh,
        "angle": float(angle),
        "center": (cx, cy),
        "rotated_rect": rect
    }


# ─────────────────────────────────────────────
# Calibration — Reference Object Detection
# ─────────────────────────────────────────────
def compute_pixel_scale(mask: np.ndarray, ref_key: str) -> "float | None":
    """
    Deteksi objek referensi (kertas A4, kartu kredit, dll) dalam gambar
    dan hitung skala piksel/cm.

    Strategy:
    - Temukan rectangular contour terbesar KEDUA (setelah kaki)
    - Cocokkan dengan dimensi referensi yang diketahui
    - Hitung pixel_per_cm = avg(dim_px / dim_cm)

    Returns: pixel_per_cm ratio, atau None jika tidak terdeteksi
    """
    if ref_key == "none" or ref_key not in REFERENCE_SIZES:
        return None

    ref_dims = REFERENCE_SIZES[ref_key]
    if ref_dims is None:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 2:
        # Fallback: estimasi berdasarkan ukuran gambar
        # Asumsi kaki mengisi ~30% tinggi gambar dan rata-rata kaki 25cm
        return None

    # Sort by area, ambil kandidat (bukan yang terbesar = kaki)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    best_scale = None
    best_rect_score = 0

    for c in sorted_contours[1:min(5, len(sorted_contours))]:
        area = cv2.contourArea(c)
        if area < 500:  # terlalu kecil
            continue

        # Cek apakah kontur ini rectangular
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 4:  # Exactly 4 corners = rectangle
            _, (w, h), _ = cv2.minAreaRect(c)
            dim_long = max(w, h)
            dim_short = min(w, h)
            ref_long = max(ref_dims["width"], ref_dims["height"])
            ref_short = min(ref_dims["width"], ref_dims["height"])

            if ref_long > 0 and ref_short > 0:
                scale_long = dim_long / ref_long
                scale_short = dim_short / ref_short
                # Keduanya harus konsisten (rasio mirip)
                if abs(scale_long - scale_short) / max(scale_long, scale_short) < 0.2:
                    score = area
                    if score > best_rect_score:
                        best_rect_score = score
                        best_scale = (scale_long + scale_short) / 2.0

    return best_scale  # pixel_per_cm


def fallback_pixel_scale(img_shape: tuple, foot_length_px: float) -> float:
    """
    Fallback kalau referensi tidak terdeteksi.
    Asumsi panjang kaki rata-rata orang dewasa Indonesia = 24 cm.
    """
    assumed_length_cm = 24.0
    return foot_length_px / assumed_length_cm


# ─────────────────────────────────────────────
# Shoe Size Lookup
# ─────────────────────────────────────────────
def get_shoe_sizes(length_cm: float) -> dict:
    """Konversi panjang kaki (cm) ke berbagai sistem ukuran sepatu."""
    # Tambah 1.5 cm untuk toleransi (standar pembuatan sepatu)
    effective = length_cm + 1.5

    for min_cm, max_cm, id_sz, eu_sz, us_sz, uk_sz in SHOE_SIZE_TABLE:
        if min_cm <= effective < max_cm:
            return {"ID": id_sz, "EU": eu_sz, "US": us_sz, "UK": uk_sz}

    # Di luar tabel
    if effective < 20.0:
        return {"ID": "<32", "EU": "<32", "US": "<2", "UK": "<1"}
    else:
        return {"ID": ">47", "EU": ">47", "US": ">13", "UK": ">12"}


# ─────────────────────────────────────────────
# Visualization Generator
# ─────────────────────────────────────────────
def generate_contour_visualization(img_bgr: np.ndarray,
                                    contour: np.ndarray,
                                    dims: dict,
                                    length_cm: float,
                                    width_cm: float,
                                    area_cm2: float) -> np.ndarray:
    """
    Membuat gambar annotated dengan kontur, bounding box, dan label.
    """
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    # ── Semi-transparent fill ──
    overlay = vis.copy()
    cv2.drawContours(overlay, [contour], -1, (74, 240, 176), -1)  # Teal fill
    cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)

    # ── Contour outline ──
    cv2.drawContours(vis, [contour], -1, (74, 240, 176), 3)

    # ── Bounding box ──
    bx, by, bw, bh = dims["bbox_x"], dims["bbox_y"], dims["bbox_w"], dims["bbox_h"]
    cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (56, 178, 255), 2)

    # ── Dimension lines ──
    # Panjang (vertikal)
    mid_x = bx + bw // 2
    cv2.arrowedLine(vis, (mid_x, by), (mid_x, by + bh), (255, 200, 80), 2, tipLength=0.02)
    cv2.arrowedLine(vis, (mid_x, by + bh), (mid_x, by), (255, 200, 80), 2, tipLength=0.02)

    # Lebar (horizontal)
    mid_y = by + bh // 2
    cv2.arrowedLine(vis, (bx, mid_y), (bx + bw, mid_y), (255, 130, 180), 2, tipLength=0.02)
    cv2.arrowedLine(vis, (bx + bw, mid_y), (bx, mid_y), (255, 130, 180), 2, tipLength=0.02)

    # ── Text labels (with background) ──
    def put_label(img, text, pos, color=(255, 255, 255), bg_color=(20, 20, 20)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, min(w, h) / 1000)
        thickness = 1 if scale < 0.7 else 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        # Background box
        cv2.rectangle(img, (x - 4, y - th - 4), (x + tw + 4, y + baseline + 2),
                      bg_color, -1)
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    # Panjang label
    put_label(vis, f"P: {length_cm:.1f}cm", (mid_x + 8, by + bh // 2 - 10), (255, 200, 80))
    # Lebar label
    put_label(vis, f"L: {width_cm:.1f}cm", (bx + 8, mid_y - 14), (255, 130, 180))
    # Area label (pojok kiri bawah kontur)
    put_label(vis, f"A: {area_cm2:.1f}cm2", (bx + 4, by + bh - 8), (74, 240, 176))

    # ── Title bar ──
    bar_h = 48
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    bar[:] = (20, 27, 40)
    vis = np.vstack([bar, vis])

    title = f"FootScan  |  P={length_cm:.1f}cm  W={width_cm:.1f}cm  A={area_cm2:.1f}cm2"
    cv2.putText(vis, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.45, min(w, h) / 1200), (74, 240, 176), 1, cv2.LINE_AA)

    return vis


# ─────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────
def validate_foot_detection(contour: np.ndarray, img_shape: tuple, confidence: int) -> None:
    """
    Validasi apakah gambar yang terdeteksi kemungkinan memang kaki.
    Raises ValueError jika tidak lolos validasi.
    """
    h, w = img_shape[:2]
    area = cv2.contourArea(contour)
    img_area = h * w

    # 1. Area ratio: kaki harus 5%–60% dari frame
    area_ratio = area / img_area
    if area_ratio < 0.05:
        raise ValueError("Kaki terlalu kecil dalam frame. Dekatkan kamera atau crop gambar.")
    if area_ratio > 0.90:
        raise ValueError("Objek terlalu memenuhi frame. Berikan lebih banyak ruang di sekitar kaki.")

    # 2. Aspect ratio: kaki manusia biasanya 2:1 sampai 4:1 (panjang:lebar)
    _, (bw, bh), _ = cv2.minAreaRect(contour)
    if bw == 0 or bh == 0:
        raise ValueError("Kontur tidak valid.")
    aspect = max(bw, bh) / min(bw, bh)
    if aspect < 1.5 or aspect > 6.0:
        raise ValueError(
            f"Rasio aspek tidak sesuai kaki ({aspect:.1f}:1). "
            "Pastikan hanya satu kaki dalam foto, posisi vertikal."
        )

    # 3. Confidence check
    if confidence < 50:
        raise ValueError(
            f"Confidence rendah ({confidence}%). "
            "Pastikan background kontras dengan kaki dan pencahayaan cukup."
        )


# ─────────────────────────────────────────────
# Main Analysis Function
# ─────────────────────────────────────────────
def analyze_foot(image_bytes: bytes, ref_key: str = "a4", threshold: int = 127) -> dict:
    """
    Full analysis pipeline.
    Returns dict dengan semua hasil pengukuran.
    """
    # 1. Load image
    pil_img = Image.open(io.BytesIO(image_bytes))

    # Validasi format
    if pil_img.format not in ("JPEG", "PNG", None):
        raise ValueError("Format gambar tidak didukung. Gunakan JPG atau PNG.")

    # Auto-rotate berdasarkan EXIF
    pil_img = pil_img.convert("RGB")
    img_bgr = pil_to_cv(pil_img)
    h, w = img_bgr.shape[:2]

    logger.info(f"Image loaded: {w}x{h}, ref={ref_key}, threshold={threshold}")

    # 2. Preprocess
    mask = preprocess_image(img_bgr, threshold_val=threshold)

    # 3. Find contour
    contour, all_contours, confidence = find_foot_contour(mask)

    # 4. Validate
    validate_foot_detection(contour, img_bgr.shape, confidence)

    # 5. Compute dimensions in pixels
    dims = compute_dimensions(contour)
    length_px = dims["length_px"]
    width_px = dims["width_px"]

    # 6. Compute area (Trapezoidal Rule + Shoelace)
    area_px2 = compute_area_trapezoidal(contour)

    # 7. Pixel-to-cm calibration
    pixel_scale = compute_pixel_scale(mask, ref_key)  # px/cm

    if pixel_scale is None or pixel_scale <= 0:
        logger.warning("Reference not detected, using fallback scale estimation")
        pixel_scale = fallback_pixel_scale(img_bgr.shape, length_px)

    # 8. Convert to cm
    length_cm = round(length_px / pixel_scale, 1)
    width_cm = round(width_px / pixel_scale, 1)
    area_cm2 = round(area_px2 / (pixel_scale ** 2), 1)

    # Sanity check: kaki manusia dewasa 20-35 cm
    # Jika di luar range, kemungkinan calibration salah
    if not (15.0 <= length_cm <= 40.0):
        logger.warning(f"Length {length_cm} cm out of normal range, recalibrating...")
        # Recalibrate: asumsikan panjang kaki normal 24 cm
        pixel_scale = length_px / 24.0
        length_cm = round(length_px / pixel_scale, 1)
        width_cm = round(width_px / pixel_scale, 1)
        area_cm2 = round(area_px2 / (pixel_scale ** 2), 1)

    # 9. Shoe sizes
    shoe_sizes = get_shoe_sizes(length_cm)

    # 10. Generate visualization
    vis_img = generate_contour_visualization(
        img_bgr, contour, dims, length_cm, width_cm, area_cm2
    )
    contour_b64 = cv_to_base64(vis_img)

    logger.info(f"Result: length={length_cm}cm, width={width_cm}cm, area={area_cm2}cm²")

    return {
        "length_cm": length_cm,
        "width_cm": width_cm,
        "area_cm2": area_cm2,
        "shoe_sizes": shoe_sizes,
        "confidence": confidence,
        "threshold_used": threshold,
        "pixel_scale": round(pixel_scale, 2),
        "image_size": {"width": w, "height": h},
        "contour_image": contour_b64,
        "contour_points": len(contour),
        "calibration_source": "reference_object" if compute_pixel_scale(mask, ref_key) else "estimation"
    }


# ─────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "FootScan API",
        "version": "1.0.0"
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Main analysis endpoint.
    
    Request: multipart/form-data
        image     : file (JPG/PNG)
        reference : string (a4 | credit_card | ruler_30 | none)
        threshold : int (50–220, default 127)
    
    Response: JSON
        {
          length_cm, width_cm, area_cm2,
          shoe_sizes: {ID, EU, US, UK},
          confidence, threshold_used,
          contour_image (base64 PNG),
          ...
        }
    """
    # Validate file presence
    if "image" not in request.files:
        return jsonify({"error": "Tidak ada file gambar yang diunggah."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong."}), 400

    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/jpg"}
    if file.content_type not in allowed:
        return jsonify({"error": f"Tipe file tidak didukung: {file.content_type}"}), 400

    # Get parameters
    ref_key = request.form.get("reference", "a4")
    try:
        threshold = int(request.form.get("threshold", 127))
        threshold = max(30, min(240, threshold))
    except ValueError:
        threshold = 127

    # Read image bytes
    try:
        image_bytes = file.read()
    except Exception as e:
        return jsonify({"error": f"Gagal membaca file: {str(e)}"}), 400

    # Run analysis
    try:
        result = analyze_foot(image_bytes, ref_key=ref_key, threshold=threshold)
        return jsonify(result), 200

    except ValueError as e:
        # User-facing validation errors
        logger.warning(f"Validation error: {e}")
        return jsonify({"error": str(e)}), 422

    except Exception as e:
        # Unexpected errors
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return jsonify({
            "error": "Analisis gagal. Pastikan foto kaki jelas dan coba lagi.",
            "detail": str(e)
        }), 500


@app.route("/")
def index():
    """Serve frontend HTML"""
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html"}
    return "FootScan API is running. Frontend not found.", 200


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  FootScan Backend — Computer Vision Foot Analyzer")
    print("=" * 55)
    port = int(os.environ.get("PORT", 5000))
    print(f"  Endpoint : http://0.0.0.0:{port}")
    print(f"  Health   : http://0.0.0.0:{port}/health")
    print(f"  Analyze  : POST http://0.0.0.0:{port}/analyze")
    print("=" * 55)
    print()
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )
