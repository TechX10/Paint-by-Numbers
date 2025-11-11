import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
import io, zipfile, cv2
from sklearn.cluster import KMeans
from reportlab.lib.pagesizes import A4, LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

st.set_page_config(page_title="Paint-by-Numbers Generator", layout="wide")
st.title("Paint-by-Numbers — Print-Ready Kit")
st.markdown("Upload to Generate to **Print & Paint!**")

def load_image(file) -> Image.Image:
    return Image.open(file).convert("RGB")

def resize_for_processing(img: Image.Image, max_dim: int = 900) -> tuple[Image.Image, float]:
    w, h = img.size
    if max(w, h) <= max_dim:
        return img.copy(), 1.0
    scale = max_dim / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    return img.resize(new_size, Image.LANCZOS), scale

# ======================
# MERGE SIMILAR COLORS
# ======================
def merge_similar_colors(centers: np.ndarray, threshold: float = 0.3) -> tuple[np.ndarray, dict]:
    if len(centers) <= 1:
        return centers, {0: 0}
    centers_f = centers.astype(np.float32)
    to_merge = list(range(len(centers)))
    merged = []
    remap = {}
    for i in to_merge:
        if i in remap:
            continue
        group = [i]
        for j in to_merge:
            if j <= i or j in remap:
                continue
            dist = np.linalg.norm(centers_f[i] - centers_f[j]) / (255 * np.sqrt(3))
            if dist < threshold:
                group.append(j)
        group_arr = centers_f[group]
        center_idx = np.argmin(np.linalg.norm(group_arr - group_arr.mean(axis=0), axis=1))
        best = group[center_idx]
        merged.append(centers[best])
        for g in group:
            remap[g] = len(merged) - 1
    return np.array(merged, dtype=np.uint8), remap

# ======================
# QUANTIZE + MERGE + CAP (UP TO 1500)
# ======================
@st.cache_data
def quantize_and_merge(
    np_img: np.ndarray, 
    max_colors: int = 24, 
    sample_fraction: float = 1.0, 
    merge_threshold: float = 0.3
):
    h, w, _ = np_img.shape
    pixels = np_img.reshape(-1, 3).astype(np.float32) / 255.0

    # Auto-sampling
    if max_colors > 500:
        sample_n = max(5000, int(pixels.shape[0] * 0.1))
    elif max_colors > 100:
        sample_n = max(5000, int(pixels.shape[0] * sample_fraction))
    else:
        sample_n = pixels.shape[0]

    if sample_n < pixels.shape[0]:
        idx = np.random.choice(pixels.shape[0], sample_n, replace=False)
        sample = pixels[idx]
    else:
        sample = pixels

    init_clusters = min(150, max_colors * 3)
    kmeans = KMeans(n_clusters=init_clusters, random_state=42, n_init=5, max_iter=300)
    kmeans.fit(sample)
    centers = (kmeans.cluster_centers_ * 255).astype(np.uint8)
    labels = kmeans.predict(pixels).reshape(h, w)

    merged_centers, merge_map = merge_similar_colors(centers, threshold=merge_threshold)
    label_map = np.vectorize(merge_map.get)(labels)

    sorted_centers, brightness_map = sort_colors_by_brightness(merged_centers)
    final_map = {old: brightness_map[merge_map.get(old, old)] for old in np.unique(label_map)}
    final_label_map = np.vectorize(final_map.get)(label_map)

    unique_labels = np.unique(final_label_map)
    if len(unique_labels) > max_colors:
        final_pixels = sorted_centers[final_label_map].reshape(-1, 3).astype(np.float32) / 255.0
        kmeans_final = KMeans(n_clusters=max_colors, random_state=42, n_init=5)
        kmeans_final.fit(final_pixels)
        final_centers = (kmeans_final.cluster_centers_ * 255).astype(np.uint8)
        final_labels = kmeans_final.labels_.reshape(h, w)
        brightness = 0.299*final_centers[:,0] + 0.587*final_centers[:,1] + 0.114*final_centers[:,2]
        order = np.argsort(brightness)
        cap_map = {i: new for new, i in enumerate(order)}
        final_label_map = np.vectorize(cap_map.get)(final_labels)
        sorted_centers = final_centers[order]
    else:
        final_centers = sorted_centers

    return final_label_map, final_centers, final_label_map.max() + 1

def tidy_small_regions(label_map: np.ndarray, min_area: int = 30) -> np.ndarray:
    label_map = label_map.copy()
    h, w = label_map.shape
    for lab in np.unique(label_map):
        mask = (label_map == lab).astype(np.uint8)
        num_cc, cc_map = cv2.connectedComponents(mask)
        for cc in range(1, num_cc):
            if (cc_map == cc).sum() < min_area:
                yx = np.column_stack(np.where(cc_map == cc))
                neigh = {}
                for y, x in yx:
                    for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < h and 0 <= nx < w and label_map[ny,nx] != lab:
                            neigh[label_map[ny,nx]] = neigh.get(label_map[ny,nx],0)+1
                if neigh:
                    best = max(neigh, key=neigh.get)
                    label_map[cc_map == cc] = best
    return label_map

def produce_outline_image(label_map: np.ndarray, thickness: int = 1, blur_sigma: float = 1.0) -> np.ndarray:
    h, w = label_map.shape
    outline = np.full((h, w, 3), 255, dtype=np.uint8)
    if blur_sigma > 0:
        blurred = cv2.GaussianBlur(label_map.astype(np.float32), (0, 0), blur_sigma)
        label_map_blurred = (blurred).astype(np.uint8)
    else:
        label_map_blurred = label_map
    for lab in np.unique(label_map):
        mask = (label_map_blurred == lab).astype(np.uint8) * 255
        edges = cv2.Canny(mask, 50, 150)
        if thickness > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
            edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(outline, contours, -1, (0, 0, 0), thickness=thickness, lineType=cv2.LINE_AA)
    return outline

def draw_one_number_per_region(outline_rgb: np.ndarray, label_map: np.ndarray, max_label_count: int = 1500,
                               font_path: str | None = None, font_pt: int = 12) -> Image.Image:
    out = Image.fromarray(outline_rgb.copy())
    draw = ImageDraw.Draw(out)
    h, w = label_map.shape
    num_labels = min(int(label_map.max()) + 1, max_label_count + 1)
    try:
        font = ImageFont.truetype(font_path, font_pt) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    placed = []
    for lab in range(num_labels):
        mask = (label_map == lab).astype(np.uint8)
        if not mask.any():
            continue
        _, cc_map = cv2.connectedComponents(mask)
        areas = [(cc, (cc_map == cc).sum()) for cc in range(1, cv2.connectedComponents(mask)[0])]
        if not areas:
            continue
        biggest_cc = max(areas, key=lambda x: x[1])[0]
        coords = np.column_stack(np.where(cc_map == biggest_cc))
        cy, cx = int(coords.mean(axis=0)[0]), int(coords.mean(axis=0)[1])
        text = str(lab + 1)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        for ox, oy in [(0,0), (tw//4,0), (-tw//4,0), (0,th//4), (0,-th//4)]:
            px = cx + ox - tw // 2
            py = cy + oy - th // 2
            if px < 0 or py < 0 or px + tw > w or py + th > h:
                continue
            box = (px, py, px + tw, py + th)
            if any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed):
                continue
            draw.text((px, py), text, fill=(0,0,0), font=font)
            placed.append(box)
            break
        else:
            px = max(0, min(w - tw, cx - tw // 2))
            py = max(0, min(h - th, cy - th // 2))
            draw.text((px, py), text, fill=(0,0,0), font=font)
            placed.append((px, py, px + tw, py + th))
    return out

def make_swatch_image(colors: list, size=(1200, 120), font_path=None, font_pt=20) -> Image.Image:
    n = len(colors)
    swatch_h = size[1]
    cell_w = size[0] // n
    sw = Image.new("RGB", (size[0], swatch_h), (255, 255, 255))
    draw = ImageDraw.Draw(sw)
    font = ImageFont.load_default() if font_path is None else ImageFont.truetype(font_path, font_pt)
    for i, col in enumerate(colors):
        if isinstance(col, (np.ndarray, np.generic)):
            col = tuple(int(c) for c in col)
        elif isinstance(col, str):
            col = ImageColor.getrgb(col)
        else:
            col = tuple(col)
        x0, x1 = i * cell_w, (i + 1) * cell_w
        draw.rectangle([x0, 0, x1, swatch_h], fill=col)
        text = str(i + 1)
        bbox = font.getbbox(text)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x0 + (cell_w - tw)//2, (swatch_h - th)//2), text, fill=(0,0,0), font=font)
    return sw

def sort_colors_by_brightness(centers: np.ndarray) -> tuple[np.ndarray, dict]:
    brightness = 0.299*centers[:,0] + 0.587*centers[:,1] + 0.114*centers[:,2]
    order = np.argsort(brightness)
    sorted_centers = centers[order]
    remap = {old: new for new, old in enumerate(order)}
    return sorted_centers, remap

def create_printable_pdf(numbered_img: Image.Image, swatch_img: Image.Image,
                         palette_text: str, page_size=A4, margin=40) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=page_size)
    width, height = page_size
    avail_w = width - 2*margin
    avail_h = height - 2*margin
    outline_h = int(avail_h * 0.70)
    img = numbered_img.copy()
    img.thumbnail((avail_w, outline_h), Image.LANCZOS)
    iw, ih = img.size
    c.drawImage(ImageReader(img), margin + (avail_w-iw)/2, height-margin-ih, width=iw, height=ih)
    sw_h = int(avail_h * 0.18)
    sw = swatch_img.copy()
    sw.thumbnail((avail_w, sw_h), Image.LANCZOS)
    sw_w, sw_h = sw.size
    yb = margin + int(avail_h*0.05)
    c.drawImage(ImageReader(sw), margin, yb, width=sw_w, height=sw_h)
    txt = c.beginText()
    txt.setTextOrigin(margin + sw_w + 20, yb + sw_h - 15)
    txt.setFont("Helvetica", 10)
    for line in palette_text.split('\n'):
        txt.textLine(line)
    c.drawText(txt)
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width/2, height-25, "Paint-by-Numbers Kit")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

uploaded = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded:
    orig = load_image(uploaded)
    proc, _ = resize_for_processing(orig, max_dim=900)
    st.image(orig, caption="Original", use_column_width=True)

    col1, col2 = st.columns([1,1])
    with col1:
        st.header("Settings")
        n_colors = st.slider("Max Color Regions", 2, 1500, 24,
                            help="Higher = more detail. Slower for >500.")
        sample_frac = st.slider("Sampling (Speed vs Accuracy)", 0.1, 1.0, 1.0, 0.05)
        color_merge = st.slider("Color Similarity Threshold", 0.1, 0.6, 0.3, 0.05,
                                help="Lower = more unique colors")
        min_region = st.slider("Min Region Area (px)", 1, 1000, 30)
        thickness = st.slider("Outline Thickness (px)", 1, 5, 1)
        smoothness = st.slider("Outline Smoothness", 0.0, 3.0, 1.0, 0.1)
        font_size = st.slider("Number Font Size (pt)", 8, 28, 14)
        page = st.selectbox("PDF Page Size", ["A4", "Letter"])

    with col2:
        st.header("Preview")
        if st.button("Generate Print-Ready Kit", type="primary"):
            np_img = np.array(proc)

            with st.spinner("Analyzing & grouping colors…"):
                final_label_map, final_centers, num_regions = quantize_and_merge(
                    np_img, max_colors=n_colors, sample_fraction=sample_frac, merge_threshold=color_merge
                )

            with st.spinner("Cleaning small regions…"):
                final_label_map = tidy_small_regions(final_label_map, min_region)

            with st.spinner("Drawing smooth outlines…"):
                outline_rgb = produce_outline_image(final_label_map, thickness=thickness, blur_sigma=smoothness)

            with st.spinner("Placing numbers…"):
                numbered = draw_one_number_per_region(outline_rgb, final_label_map, font_pt=font_size)

            with st.spinner("Building swatch…"):
                swatch = make_swatch_image(
                    final_centers.tolist(),
                    size=(min(2400, max(800, 40 * num_regions)), 80),
                    font_pt=16
                )

            st.image(numbered, caption=f"Numbered Outline ({num_regions} regions)", use_column_width=True)
            st.image(swatch, caption="Color Swatch", use_column_width=True)

            palette_lines = [f"{i+1}: rgb({r},{g},{b})  #{r:02x}{g:02x}{b:02x}"
                            for i, (r,g,b) in enumerate(final_centers)]
            palette_text = "\n".join(palette_lines)

            page_sz = A4 if page == "A4" else LETTER
            with st.spinner("Creating PDF…"):
                pdf_bytes = create_printable_pdf(numbered, swatch, palette_text, page_sz)

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, img in [("pbn_numbered.png", numbered),
                                  ("pbn_outline.png", Image.fromarray(outline_rgb)),
                                  ("pbn_swatch.png", swatch)]:
                    b = io.BytesIO()
                    img.save(b, "PNG")
                    zf.writestr(name, b.getvalue())
                zf.writestr("pbn_printable.pdf", pdf_bytes)
                zf.writestr("palette.txt", palette_text.encode())
            zip_buf.seek(0)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("Download PDF (Print!)", pdf_bytes,
                                   "paint_by_numbers.pdf", "application/pdf")
            with c2:
                st.download_button("Download Full ZIP", zip_buf,
                                   "pbn_kit.zip", "application/zip")

            with st.expander("Palette Details"):
                st.code(palette_text)

    st.markdown("""
    ### Print Instructions
    1. **Download PDF** to 2. Print on **A4/Letter** to 3. **"Actual Size"** to 4. Paint!
    """)