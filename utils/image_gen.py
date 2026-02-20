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


def _detect_character_name(story_content: str, character_type: str) -> str:
    """
    Try to extract the main character's name from the story.
    e.g. 'Leo the lion', 'Bella the rabbit', 'Max the dog'.
    Falls back to a generic description if no name is found.
    """
    # Look for patterns like 'Name the animal' or 'Name, a/an animal'
    patterns = [
        rf"([A-Z][a-z]+)(?:,? (?:the|a|an) {character_type})",
        rf"([A-Z][a-z]+)(?:'s| was| had| lived| went| said| wanted)",
    ]
    for pat in patterns:
        m = re.search(pat, story_content)
        if m:
            name = m.group(1)
            # Filter common non-name words
            if name.lower() not in {"once","one","in","the","a","an","there","he","she","it"}:
                return f"{name} the {character_type}"
    return f"the {character_type}"


# ── Scene splitting ────────────────────────────────────────────────────────────
_STAGE_LABELS = ["Scene 1", "Scene 2", "Scene 3", "Scene 4"]

def _split_scenes(story: str, n: int = 4) -> list:
    """Split story into exactly n non-overlapping sequential scenes."""
    # Split on sentence boundaries
    sents = re.split(r"(?<=[.!?])\s+", story.strip())
    sents = [s.strip() for s in sents if s.strip()]
    # Pad if too few sentences
    while len(sents) < n:
        sents.append(sents[-1])
    # Distribute sentences as evenly as possible across n panels
    base, extra = divmod(len(sents), n)
    scenes = []
    start = 0
    for i in range(n):
        end = start + base + (1 if i < extra else 0)
        scenes.append(" ".join(sents[start:end])[:1000])  # no hard cut—keep full scene
        start = end
    return scenes


# ── AI prompt builder ──────────────────────────────────────────────────────────
_STAGE_MOOD = [
    "peaceful happy morning, introduction of the hero",
    "tense dramatic moment, rising action and challenge",
    "brave heroic adventure, climax and problem-solving",
    "joyful celebration, happy ending and resolution",
]

def _build_prompt(scene_text: str, character_type: str, stage_idx: int,
                  character_name: str = "", prev_scene: str = "") -> str:
    """
    Build an image prompt that:
    - Always revolves around the named main character
    - Reflects the mood/stage of the story
    - Includes a brief summary of the previous scene for visual continuity
    """
    clean      = re.sub(r"\*+", "", scene_text).strip()[:200]
    mood       = _STAGE_MOOD[stage_idx % len(_STAGE_MOOD)]
    char_desc  = character_name if character_name else f"cute {character_type}"

    # Continuity context from the previous panel
    if prev_scene:
        prev_clean = re.sub(r"\*+", "", prev_scene).strip()[:100]
        continuation = f"continuing the story from previous scene: '{prev_clean}', now: "
    else:
        continuation = "story begins: "

    return (
        f"children's storybook comic panel, {continuation}"
        f"{clean}. "
        f"Main character: {char_desc}, same adorable character in every panel, "
        f"{mood}, Pixar Disney cartoon style, "
        f"vibrant colors, expressive face, kids comic book art, "
        f"clean bold outlines, digital painting, high quality, no text, no watermark"
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


# ── Overlay: add scene badge + full caption on top of any image ───────────────
def _add_overlay(image_bytes: bytes, scene_text: str,
                 panel_idx: int, size=(640, 800)) -> bytes:
    """
    Compose a tall image:
      - Top portion (640x512): the AI/fallback illustration
      - Bottom caption zone: full scene text, never truncated
    The output is 640 wide x (512 + caption_height) tall.
    """
    from PIL import Image, ImageDraw, ImageFont

    ILL_W, ILL_H = 640, 512          # illustration area
    SIDE_PAD     = 16
    LINE_H       = 20
    TOP_PAD      = 12
    BOT_PAD      = 14
    TEXT_W       = ILL_W - SIDE_PAD * 2
    WRAP_CHARS   = TEXT_W // 8       # ~76 chars at font size 14

    acc    = _hex(ACCENTS[panel_idx % len(ACCENTS)])
    labels = ["📖 Scene 1 of 4", "📖 Scene 2 of 4",
               "📖 Scene 3 of 4", "📖 Scene 4 of 4"]

    try:
        fB = ImageFont.truetype("arialbd.ttf", 18)
        fC = ImageFont.truetype("arial.ttf",   15)
    except Exception:
        fB = fC = ImageFont.load_default()

    # -- illustration --
    ill = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(
        (ILL_W, ILL_H), Image.LANCZOS)

    # Scene badge on illustration (top-left)
    draw_ill = ImageDraw.Draw(ill, "RGBA")
    bw, bh = 200, 30
    draw_ill.rounded_rectangle([8, 8, 8+bw, 8+bh], radius=8, fill=(*acc, 220))
    draw_ill.text((8+bw//2, 8+bh//2), labels[panel_idx],
                  fill="white", font=fB, anchor="mm")
    draw_ill.rectangle([0, 0, ILL_W-1, ILL_H-1], outline="#1A1A1A", width=4)

    # -- caption zone (full text, never cut) --
    clean_text = re.sub(r"\*+", "", scene_text).strip()
    cap_lines  = textwrap.wrap(clean_text, width=WRAP_CHARS) or [""]
    cap_h      = TOP_PAD + len(cap_lines) * LINE_H + BOT_PAD

    # Build final canvas
    canvas = Image.new("RGB", (ILL_W, ILL_H + cap_h), "#FFFBF0")
    canvas.paste(ill, (0, 0))

    # Caption background
    draw_cap = ImageDraw.Draw(canvas, "RGBA")
    draw_cap.rectangle([0, ILL_H, ILL_W, ILL_H + cap_h], fill=(20, 20, 20, 255))
    draw_cap.rectangle([0, ILL_H, ILL_W, ILL_H + 5],     fill=(*acc, 255))

    for li, ln in enumerate(cap_lines):
        draw_cap.text(
            (ILL_W // 2, ILL_H + TOP_PAD + li * LINE_H),
            ln, fill="white", font=fC, anchor="mm"
        )

    # Outer border
    draw_cap.rectangle([0, 0, ILL_W-1, ILL_H+cap_h-1], outline="#1A1A1A", width=4)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


# _compose_page removed — each scene is now saved as its own image file.


# ── Public API ─────────────────────────────────────────────────────────────────
def generate_comic_panels(story_id: int, story_content: str,
                           story_title: str = "") -> list:
    """
    Generate 4 individual comic images — one per scene.
    Each image = AI illustration (640x512) + full scene text caption below it.
    Returns a list of 4 file paths:
        ['generated_comics/story_1_scene_1.png', ..._scene_4.png]
    """
    t0 = time.time()
    character_type = _detect_character(story_content)
    print(f"[image_gen] Character detected: {character_type}")

    scenes = _split_scenes(story_content, 4)
    character_name = _detect_character_name(story_content, character_type)
    print(f"[image_gen] Main character: {character_name}")
    print(f"[image_gen] Scenes:\n" +
          "\n".join(f"  Scene {i+1}: {s[:80]}..." for i, s in enumerate(scenes)))

    # Build prompts — named character + previous scene context
    prompts = [
        _build_prompt(
            scene_text=scenes[i],
            character_type=character_type,
            stage_idx=i,
            character_name=character_name,
            prev_scene=scenes[i - 1] if i > 0 else "",
        )
        for i in range(len(scenes))
    ]

    # ── Parallel AI image fetch ───────────────────────────────────────────────
    ai_images: list = [None] * 4
    source = "HuggingFace" if HF_TOKEN else "Pollinations"
    print(f"[image_gen] Fetching 4 AI images in parallel (primary: {source})...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        future_map = {pool.submit(_fetch_ai_image, p, i): i
                      for i, p in enumerate(prompts)}
        for fut in as_completed(future_map):
            ai_images[future_map[fut]] = fut.result()

    # ── Build and save each panel as a separate file ──────────────────────────
    saved_paths = []
    for i, (ai_bytes, scene) in enumerate(zip(ai_images, scenes)):
        if ai_bytes:
            panel_bytes = _add_overlay(ai_bytes, scene, i)
        else:
            print(f"[image_gen] Panel {i+1}: using Pillow fallback")
            fallback = _make_fallback_panel(scene, i, character_type)
            panel_bytes = _add_overlay(fallback, scene, i)

        path = f"{OUTPUT_DIR}/story_{story_id}_scene_{i+1}.png"
        with open(path, "wb") as f:
            f.write(panel_bytes)
        saved_paths.append(path)
        print(f"[image_gen] Saved scene {i+1} → {path}")

    elapsed = time.time() - t0
    ai_count = sum(1 for x in ai_images if x)
    print(f"[image_gen] Done in {elapsed:.1f}s — {ai_count}/4 AI, "
          f"{4-ai_count}/4 fallback.")
    return saved_paths


# Keep old names as aliases for backwards compatibility
def generate_full_comic_page(story_id: int, story_content: str,
                              story_title: str = "") -> list:
    """Alias for generate_comic_panels (now returns list of 4 paths)."""
    return generate_comic_panels(story_id, story_content, story_title)


def generate_comic_images(story_id: int, story_content: str) -> list:
    """Legacy wrapper."""
    return generate_comic_panels(story_id, story_content)
