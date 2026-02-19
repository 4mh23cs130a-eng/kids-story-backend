"""
image_gen.py — Comic page generator with REAL AI images.

Strategy:
  1. Build one AI prompt per scene (with character type + story context)
  2. Fetch all 4 images IN PARALLEL from Pollinations.ai  (free, no key)
  3. If any panel's AI fetch fails fall back to Pillow-drawn illustration
  4. Compose the 4 panel images into a single comic page with:
       - Gold title header
       - Thick comic borders
       - Scene number badge
       - Story-text caption bar
"""

import os, io, re, math, random, textwrap, time, requests
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
OUTPUT_DIR = "generated_comics"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ── Pollinations ──────────────────────────────────────────────────────────────
POLL_URL = (
    "https://image.pollinations.ai/prompt/{prompt}"
    "?width=512&height=512&nologo=true&seed={seed}"
)

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_TOKEN   = os.getenv("HF_TOKEN", "").strip()
HF_MODEL   = "black-forest-labs/FLUX.1-schnell"   # fast, open-weight
HF_URL     = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL}"

# ── Accent colours (one per panel) ────────────────────────────────────────────
ACCENTS = ["#E53935", "#1565C0", "#E65100", "#6A1B9A"]
PALETTES = [
    {"sky1":"#87CEEB","sky2":"#C8E6FF","hill":"#A5D6A7","ground":"#66BB6A",
     "accent":"#E53935","char":"#FF8F00","flower":"#F48FB1"},
    {"sky1":"#B0BEC5","sky2":"#CFD8DC","hill":"#81C784","ground":"#558B2F",
     "accent":"#1565C0","char":"#6A1B9A","flower":"#CE93D8"},
    {"sky1":"#FFD54F","sky2":"#FFF9C4","hill":"#C5E1A5","ground":"#7CB342",
     "accent":"#E65100","char":"#00695C","flower":"#FFCC02"},
    {"sky1":"#7E57C2","sky2":"#B39DDB","hill":"#AED581","ground":"#8BC34A",
     "accent":"#880E4F","char":"#1A237E","flower":"#FFD600"},
]


# ── Character detection ────────────────────────────────────────────────────────
CHARACTER_KEYWORDS = {
    "lion":     ["lion","lioness","cub","leo"],
    "bear":     ["bear","panda","polar bear","teddy"],
    "cat":      ["cat","kitten","kitty"],
    "dog":      ["dog","puppy","pup","hound"],
    "elephant": ["elephant","jumbo","ellie"],
    "fox":      ["fox","vixen"],
    "monkey":   ["monkey","ape","chimp"],
    "bird":     ["bird","parrot","owl","eagle","duck","crow","sparrow"],
    "tiger":    ["tiger","tigress"],
    "deer":     ["deer","fawn","stag"],
    "rabbit":   ["rabbit","bunny","hare"],
}

def _detect_character(story_content: str) -> str:
    txt = story_content.lower()
    for ctype, kws in CHARACTER_KEYWORDS.items():
        if any(kw in txt for kw in kws):
            return ctype
    return "rabbit"


# ── Scene splitting ────────────────────────────────────────────────────────────
_STAGE_LABELS = ["Beginning", "Conflict", "Action", "Resolution"]

def _split_scenes(story: str, n: int = 4) -> list:
    sents = re.split(r"(?<=[.!?])\s+", story.strip())
    sents = [s.strip() for s in sents if s.strip()]
    while len(sents) < n:
        sents.append(sents[-1])
    chunk = max(1, len(sents) // n)
    scenes = []
    for i in range(n):
        s = i * chunk
        e = s + chunk if i < n - 1 else len(sents)
        scenes.append(" ".join(sents[s:e])[:280])
    return scenes


# ── AI prompt builder ──────────────────────────────────────────────────────────
_STAGE_MOOD = [
    "happy peaceful morning",
    "stormy dangerous dramatic",
    "brave heroic adventure",
    "joyful celebration colorful",
]

def _build_prompt(scene_text: str, character_type: str, stage_idx: int) -> str:
    clean = re.sub(r"\*+", "", scene_text).strip()
    mood  = _STAGE_MOOD[stage_idx % len(_STAGE_MOOD)]
    return (
        f"children's storybook illustration, cute {character_type} character, "
        f"{mood}, {clean}, "
        f"Pixar Disney cartoon style, vibrant colors, colorful background, "
        f"kids comic book art, clean outlines, digital painting, "
        f"high quality, detailed, no text, no watermark"
    )


# ── AI image fetch: HuggingFace → Pollinations → None ─────────────────────────
def _fetch_hf(prompt: str, panel_idx: int, timeout: int = 40) -> bytes | None:
    """Try HuggingFace FLUX.1-schnell. Returns None if token lacks permission."""
    if not HF_TOKEN:
        return None
    try:
        print(f"[image_gen] Panel {panel_idx+1} → HuggingFace FLUX...")
        resp = requests.post(
            HF_URL,
            headers={"Authorization": f"Bearer {HF_TOKEN}",
                     "Content-Type": "application/json"},
            json={"inputs": prompt,
                  "parameters": {"width": 512, "height": 512}},
            timeout=timeout,
        )
        if resp.status_code == 200 and len(resp.content) > 5000:
            print(f"[image_gen] Panel {panel_idx+1} ✓ HF AI image ({len(resp.content)//1024} KB)")
            return resp.content
        print(f"[image_gen] Panel {panel_idx+1} HF HTTP {resp.status_code}: {resp.text[:80]}")
    except Exception as e:
        print(f"[image_gen] Panel {panel_idx+1} HF error: {e}")
    return None


def _fetch_pollinations(prompt: str, panel_idx: int, timeout: int = 28) -> bytes | None:
    """Try Pollinations.ai as backup."""
    url = POLL_URL.format(prompt=quote(prompt), seed=panel_idx * 42 + 7)
    try:
        print(f"[image_gen] Panel {panel_idx+1} → Pollinations...")
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 5000:
            print(f"[image_gen] Panel {panel_idx+1} ✓ Pollinations AI image ({len(resp.content)//1024} KB)")
            return resp.content
        print(f"[image_gen] Panel {panel_idx+1} Pollinations HTTP {resp.status_code}")
    except Exception as e:
        print(f"[image_gen] Panel {panel_idx+1} Pollinations error: {e}")
    return None


def _fetch_ai_image(prompt: str, panel_idx: int) -> bytes | None:
    """HuggingFace first, Pollinations fallback."""
    result = _fetch_hf(prompt, panel_idx)
    if result:
        return result
    return _fetch_pollinations(prompt, panel_idx)



# ── Pillow fallback (single panel) ────────────────────────────────────────────
def _hex(h):
    if isinstance(h, (tuple, list)):
        return tuple(int(c) for c in h[:3])
    c = h.lstrip("#")
    return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))

def _lerp(c1, c2, t):
    r1,g1,b1 = _hex(c1); r2,g2,b2 = _hex(c2)
    return (int(r1+(r2-r1)*t), int(g1+(g2-g1)*t), int(b1+(b2-b1)*t))

def _lighten(c, a=55):
    r,g,b = _hex(c); return (min(255,r+a),min(255,g+a),min(255,b+a))
def _darken(c, a=40):
    r,g,b = _hex(c); return (max(0,r-a),max(0,g-a),max(0,b-a))

def _grad(draw, x, y, w, h, c1, c2):
    for row in range(h):
        t = row / max(h-1,1)
        draw.line([(x,y+row),(x+w,y+row)], fill=_lerp(c1,c2,t))

def _cloud(draw, cx, cy, s=1.0):
    for dx,dy,r in [(-22,2,20),(0,0,27),(22,2,20),(-12,-13,18),(12,-11,18)]:
        sx,sy,sr = int(dx*s),int(dy*s),int(r*s)
        draw.ellipse([cx+sx-sr,cy+sy-sr,cx+sx+sr,cy+sy+sr],fill=(255,255,255,210))

def _tree(draw, x, y, h=65):
    draw.rectangle([x-5,y-h//5,x+5,y],fill="#5D4037")
    for ow,off in [(44,0),(34,h//5),(26,h//3+2)]:
        ty=y-h//5-off
        draw.polygon([(x,ty-h//2+off),(x-ow//2,ty),(x+ow//2,ty)],
                     fill="#388E3C",outline="#1B5E20",width=1)

def _sun(draw, cx, cy, r=22):
    draw.ellipse([cx-r,cy-r,cx+r,cy+r],fill="#FFD740",outline="#FFA000",width=2)
    for a in range(0,360,30):
        rd=math.radians(a)
        draw.line([(cx+int((r+4)*math.cos(rd)),cy+int((r+4)*math.sin(rd))),
                   (cx+int((r+16)*math.cos(rd)),cy+int((r+16)*math.sin(rd)))],
                  fill="#FFC107",width=3)

def _hill(draw, x, y, w, h, col):
    pts=[(x,y+h),(x+w,y+h)]
    for i in range(21):
        a=math.pi*i/20
        pts.insert(-1,(x+int(w*i/20),y+h-int(h*math.sin(a))))
    draw.polygon(pts,fill=col)

def _flower(draw, cx, cy, col="#F48FB1", sz=7):
    for a in range(0,360,60):
        rd=math.radians(a)
        draw.ellipse([cx+int(sz*math.cos(rd))-4,cy+int(sz*math.sin(rd))-4,
                      cx+int(sz*math.cos(rd))+4,cy+int(sz*math.sin(rd))+4],fill=col)
    draw.ellipse([cx-3,cy-3,cx+3,cy+3],fill="#FFFF00")

def _char_blob(draw, cx, cy, col, sz, character_type):
    """Draw a simple but recognizable character blob as fallback."""
    hs = int(sz*0.54)
    hx,hy = cx, cy-sz//2-hs+6
    # body
    draw.ellipse([cx-sz//2+3,cy-sz//2+3,cx+sz//2+3,cy+sz//2+3],fill=(0,0,0,50))
    draw.ellipse([cx-sz//2,cy-sz//2,cx+sz//2,cy+sz//2],fill=col,outline="#1A1A1A",width=2)
    # head
    draw.ellipse([hx-hs,hy-hs,hx+hs,hy+hs],fill=col,outline="#1A1A1A",width=2)
    # ears / head details by type
    if character_type in ("rabbit","bunny"):
        for ex in [hx-hs*2//3,hx+hs//4]:
            ew=max(10,hs//2-2)
            draw.ellipse([ex,hy-hs*2,ex+ew,hy-hs+8],fill=col,outline="#1A1A1A",width=2)
            if ew>8: draw.ellipse([ex+3,hy-hs*2+5,ex+ew-3,hy-hs+10],fill="#FFB3C1")
    elif character_type == "lion":
        mane="#E65100"
        for ma in range(0,360,30):
            mr=math.radians(ma)
            mx=hx+int((hs+8)*math.cos(mr)); my=hy+int((hs+8)*math.sin(mr))
            draw.ellipse([mx-9,my-9,mx+9,my+9],fill=mane)
        draw.ellipse([hx-hs,hy-hs,hx+hs,hy+hs],fill=col,outline="#1A1A1A",width=2)
        for ex2 in [hx-hs+2,hx+hs-14]:
            draw.ellipse([ex2-2,hy-hs-6,ex2+14,hy-hs+8],fill=col,outline="#1A1A1A",width=2)
    elif character_type == "elephant":
        for ex_d in [-1,1]:
            ear_cx=hx+ex_d*(hs+12)
            draw.ellipse([ear_cx-14,hy-hs+4,ear_cx+14,hy+hs-4],
                         fill=_lighten(col,15),outline="#1A1A1A",width=2)
        pts=[(hx+int((hs-4)*math.cos(math.radians(-90+t*16)+0.4)),
              hy+hs-2+t*4) for t in range(11)]
        if len(pts)>=2: draw.line(pts,fill=col,width=7)
    elif character_type in ("cat","fox"):
        for ex_d in [-1,1]:
            epx=hx+ex_d*(hs-4)
            draw.polygon([(epx-10,hy-hs+4),(epx+10,hy-hs+4),(epx,hy-hs-18)],
                         fill=col,outline="#1A1A1A",width=2)
    elif character_type == "dog":
        for ex_d in [-1,1]:
            eax=hx+ex_d*(hs-2)
            draw.ellipse([eax-8,hy-hs//2,eax+8,hy+hs+10],
                         fill=_darken(col,30),outline="#1A1A1A",width=2)
    else:  # bear, monkey, deer, bird, tiger, default
        for ex2 in [hx-hs-2,hx+hs-16]:
            draw.ellipse([ex2,hy-hs-8,ex2+20,hy-hs+10],fill=col,outline="#1A1A1A",width=2)
    # eyes
    ew=max(3,hs//5)
    for eox in [-hs//3,hs//3]:
        draw.ellipse([hx+eox-ew,hy-ew,hx+eox+ew,hy+ew],fill="#1A1A1A")
        draw.ellipse([hx+eox+ew//3,hy-ew+1,hx+eox+ew,hy-ew//2],fill="white")
    draw.arc([hx-hs//2+2,hy+2,hx+hs//2-2,hy+hs//2+2],0,180,fill="#1A1A1A",width=2)

def _make_fallback_panel(scene_text: str, panel_idx: int,
                          character_type: str, size=(512,512)) -> bytes:
    """Generate a nice illustrated Pillow panel when AI is unavailable."""
    from PIL import Image, ImageDraw
    W, H = size
    pal = PALETTES[panel_idx % len(PALETTES)]
    txt = scene_text.lower()
    is_storm = any(w in txt for w in ["storm","rain","scared","danger","dark"])

    img  = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img, "RGBA")
    rng  = random.Random(panel_idx * 13)

    gnd_y = int(H * 0.60)
    # Sky gradient
    _grad(draw, 0, 0, W, gnd_y, pal["sky1"], pal["sky2"])
    # Hills
    _hill(draw, 0, int(H*0.44), W, int(H*0.18),
          _lerp(pal["hill"],"#FFFFFF",0.35))
    # Ground
    _grad(draw, 0, gnd_y, W, H-gnd_y, pal["ground"], _darken(pal["ground"],30))
    # Grass tufts
    for gx in range(4, W-4, 14):
        gy = gnd_y + rng.randint(-2,2)
        draw.line([(gx,gy),(gx-3,gy-8)],fill="#33691E",width=2)
        draw.line([(gx,gy),(gx+3,gy-8)],fill="#2E7D32",width=2)
    # Weather
    if is_storm:
        for cx,cy,cs in [(W//4,30,1.6),(W//2,16,2.0),(3*W//4,34,1.5)]:
            _cloud(draw,cx,cy,cs)
            draw.ellipse([cx-int(28*cs),cy-int(18*cs),cx+int(28*cs),cy+int(18*cs)],
                         fill=(130,130,140,180))
        for _ in range(24):
            rx=rng.randint(4,W-4); ry=rng.randint(35,int(H*0.55))
            draw.line([(rx,ry),(rx-4,ry+14)],fill=(80,130,200,190),width=2)
        lx=W//2
        draw.line([(lx,20),(lx-10,50),(lx+4,50),(lx-12,90)],fill="#FFD740",width=4)
    else:
        _sun(draw, W-45, 32, r=22)
        _cloud(draw, W//5, 24, 1.1)
        _cloud(draw, W//2+10, 34, 0.9)
    # Trees
    _tree(draw, 22,  gnd_y, h=62)
    _tree(draw, W-26, gnd_y, h=55)
    # Flowers
    for fx,fy in [(58,gnd_y+2),(95,gnd_y+4),(W-62,gnd_y+3),(W-98,gnd_y+6)]:
        _flower(draw, fx,fy, pal["flower"])
    # Path
    path_pts=[(W//2-20,gnd_y+5),(W//2-32,H-4),(W//2+32,H-4),(W//2+20,gnd_y+5)]
    draw.polygon(path_pts, fill=_lerp(pal["ground"],"#FFFFFF",0.45))
    # Characters
    cy2 = gnd_y - 4
    _char_blob(draw, W//2-18, cy2, pal["char"],   40, character_type)
    friend_type = "rabbit" if character_type != "rabbit" else "bear"
    _char_blob(draw, W//2+46, cy2+6, "#EF9A9A", 28, friend_type)

    buf = io.BytesIO(); img.save(buf,format="PNG"); return buf.getvalue()


# ── Overlay: add scene badge + caption on top of any image ────────────────────
def _add_overlay(image_bytes: bytes, scene_text: str,
                 panel_idx: int, size=(512,512)) -> bytes:
    """
    Add a semi-transparent scene badge and caption bar on top of an image.
    Works on both AI images and Pillow fallback panels.
    """
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(size, Image.LANCZOS)
    draw = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    acc = _hex(ACCENTS[panel_idx % len(ACCENTS)])
    labels = ["① Beginning","② Conflict","③ Action","④ Resolution"]

    try:
        fB = ImageFont.truetype("arialbd.ttf", 16)
        fC = ImageFont.truetype("arial.ttf",   14)
    except Exception:
        fB = fC = ImageFont.load_default()

    # Scene badge (top-left)
    bw, bh = 148, 28
    draw.rounded_rectangle([8,8,8+bw,8+bh], radius=8, fill=(*acc,220))
    draw.text((8+bw//2,8+bh//2), labels[panel_idx], fill="white", font=fB, anchor="mm")

    # Caption bar (bottom)
    cap_lines = textwrap.wrap(re.sub(r"\*+","",scene_text).strip(), width=46)[:3]
    cap_h = len(cap_lines)*19 + 18
    draw.rectangle([0,H-cap_h,W,H], fill=(10,10,10,215))
    draw.rectangle([0,H-cap_h,W,H-cap_h+4], fill=(*acc,255))
    for li,ln in enumerate(cap_lines):
        draw.text((W//2, H-cap_h+10+li*19), ln, fill="white", font=fC, anchor="mm")

    # Panel border
    draw.rectangle([0,0,W-1,H-1], outline="#1A1A1A", width=4)

    buf = io.BytesIO(); img.save(buf,format="PNG"); return buf.getvalue()


# ── Comic page compositor ──────────────────────────────────────────────────────
def _compose_page(panel_images: list, scene_texts: list,
                  story_title: str) -> bytes:
    """Compose 4 panel images (512×512 each) into one comic page."""
    from PIL import Image, ImageDraw, ImageFont

    PANEL_W, PANEL_H = 512, 512
    GAP    = 12
    BORDER = 6
    HDR_H  = 78
    PAGE_W = PANEL_W*2 + GAP*3
    PAGE_H = HDR_H + PANEL_H*2 + GAP*3

    page = Image.new("RGB", (PAGE_W, PAGE_H), "#FFFBF0")
    draw = ImageDraw.Draw(page, "RGBA")

    # Dot-grid texture
    for dy in range(0, PAGE_H, 16):
        for dx in range(0, PAGE_W, 16):
            draw.ellipse([dx-2,dy-2,dx+2,dy+2], fill=(0,0,0,16))

    try:
        fT = ImageFont.truetype("arialbd.ttf", 28)
    except Exception:
        fT = ImageFont.load_default()

    # Header
    _grad(draw, 0, 0, PAGE_W, HDR_H, "#212121", "#424242")
    draw.rectangle([3,3,PAGE_W-3,HDR_H-3], outline="#F9A825", width=3)
    _grad(draw, 5, 5, PAGE_W-10, HDR_H-10, "#F9A825", "#FFC400")
    clean = re.sub(r"\*+","",story_title).strip() or "Kids Story"
    draw.text((PAGE_W//2+2,HDR_H//2+2), f"✦ {clean.upper()} ✦",
              fill=(0,0,0,100), font=fT, anchor="mm")
    draw.text((PAGE_W//2,HDR_H//2), f"✦ {clean.upper()} ✦",
              fill="#1A237E", font=fT, anchor="mm")

    positions = [
        (GAP,             HDR_H+GAP),
        (GAP*2+PANEL_W,   HDR_H+GAP),
        (GAP,             HDR_H+GAP*2+PANEL_H),
        (GAP*2+PANEL_W,   HDR_H+GAP*2+PANEL_H),
    ]

    for i, (px, py) in enumerate(positions):
        acc = _hex(ACCENTS[i])
        # Coloured outer frame
        draw.rectangle([px-4,py-4,px+PANEL_W+4,py+PANEL_H+4],
                       fill=ACCENTS[i], outline="#1A1A1A", width=2)
        # Paste panel image
        if i < len(panel_images) and panel_images[i]:
            try:
                pimg = Image.open(io.BytesIO(panel_images[i])).convert("RGB")
                pimg = pimg.resize((PANEL_W, PANEL_H), Image.LANCZOS)
                page.paste(pimg, (px, py))
            except Exception as e:
                print(f"[image_gen] Paste error panel {i}: {e}")
                page.paste(Image.new("RGB",(PANEL_W,PANEL_H),"#EEE"), (px,py))
        # Panel border
        draw.rectangle([px,py,px+PANEL_W-1,py+PANEL_H-1], outline="#1A1A1A", width=BORDER)

    # Page border
    draw.rectangle([0,0,PAGE_W-1,PAGE_H-1], outline="#1A1A1A", width=6)

    buf = io.BytesIO(); page.save(buf, format="PNG"); return buf.getvalue()


# ── Public API ─────────────────────────────────────────────────────────────────
def generate_full_comic_page(story_id: int, story_content: str,
                              story_title: str = "") -> str:
    """
    Generate a full 4-panel comic page using REAL AI images from Pollinations.ai.
    All 4 images are fetched in parallel (~15-25s).
    Falls back to Pillow illustration if AI is unavailable.
    """
    t0 = time.time()
    character_type = _detect_character(story_content)
    print(f"[image_gen] Character detected: {character_type}")

    scenes  = _split_scenes(story_content, 4)
    prompts = [_build_prompt(s, character_type, i) for i, s in enumerate(scenes)]

    # ── Parallel AI fetch (HF → Pollinations → Pillow) ──────────────────────────
    ai_images = [None] * 4
    source = "HuggingFace" if HF_TOKEN else "Pollinations"
    print(f"[image_gen] Fetching {len(prompts)} AI images in parallel (primary: {source})...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {pool.submit(_fetch_ai_image, p, i): i
                      for i, p in enumerate(prompts)}

        for fut in as_completed(future_map):
            idx = future_map[fut]
            ai_images[idx] = fut.result()

    # ── Build panel images (AI + overlay, or Pillow fallback) ────────────────
    panel_images = []
    for i, (ai_bytes, scene) in enumerate(zip(ai_images, scenes)):
        if ai_bytes:
            panel_bytes = _add_overlay(ai_bytes, scene, i)
        else:
            print(f"[image_gen] Panel {i+1}: using Pillow fallback")
            fallback = _make_fallback_panel(scene, i, character_type)
            panel_bytes = _add_overlay(fallback, scene, i)
        panel_images.append(panel_bytes)

    # ── Compose page ────────────────────────────────────────────────────────
    page_bytes = _compose_page(panel_images, scenes, story_title)
    out = f"{OUTPUT_DIR}/story_{story_id}.png"
    with open(out, "wb") as f:
        f.write(page_bytes)

    elapsed = time.time() - t0
    ai_count = sum(1 for x in ai_images if x)
    print(f"[image_gen] Done in {elapsed:.1f}s — {ai_count}/4 AI images, "
          f"{4-ai_count}/4 fallback. Saved: {out}")
    return out


def generate_comic_images(story_id: int, story_content: str) -> list:
    """Legacy wrapper."""
    return [generate_full_comic_page(story_id, story_content)]
