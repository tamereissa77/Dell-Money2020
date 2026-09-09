"""Generate the Money20/20 Middle East deck for the GB10 fraud demo.
Rebuild with:  ./.venv/bin/python deck/build_deck.py
"""
import pathlib
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

C_BG    = RGBColor(0x0A,0x0E,0x13)
C_PANEL = RGBColor(0x14,0x1C,0x25)
C_GREEN = RGBColor(0x76,0xB9,0x00)
C_RED   = RGBColor(0xFF,0x57,0x47)
C_AMBER = RGBColor(0xFF,0xA5,0x02)
C_TEXT  = RGBColor(0xE8,0xEE,0xF5)
C_DIM   = RGBColor(0x8B,0xA0,0xB6)
C_LINE  = RGBColor(0x22,0x30,0x3F)
C_DELL  = RGBColor(0x00,0x76,0xCE)
C_DELLL = RGBColor(0x4A,0xA8,0xE8)
SANS, MONO = "Segoe UI", "Consolas"
W, H = Inches(13.333), Inches(7.5)

prs = Presentation(); prs.slide_width, prs.slide_height = W, H
blank = prs.slide_layouts[6]

def slide(rule=True):
    s = prs.slides.add_slide(blank)
    bg = s.shapes.add_shape(1, 0, 0, W, H)
    bg.fill.solid(); bg.fill.fore_color.rgb = C_BG; bg.line.fill.background()
    bg.shadow.inherit = False
    if rule: brand_rule(s)
    return s

def tb(s, x, y, w, h, align=PP_ALIGN.LEFT):
    t = s.shapes.add_textbox(x, y, w, h); tf = t.text_frame
    tf.word_wrap = True; tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    tf.paragraphs[0].alignment = align
    return tf

def line(tf, text, size, color, bold=False, font=SANS, space_after=6, first=False, align=None):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.text = text
    p.space_after = Pt(space_after)
    if align: p.alignment = align
    targets = [r.font for r in p.runs] or [p.font]
    for f in targets:
        f.size, f.color.rgb, f.bold, f.name = Pt(size), color, bold, font
    return p

def rule(s, x, y, w, color=C_GREEN, h=Emu(28575)):
    r = s.shapes.add_shape(1, x, y, w, h)
    r.fill.solid(); r.fill.fore_color.rgb = color; r.line.fill.background()
    r.shadow.inherit = False; return r

def panel(s, x, y, w, h, color=C_PANEL):
    p = s.shapes.add_shape(5, x, y, w, h)
    p.fill.solid(); p.fill.fore_color.rgb = color
    p.line.color.rgb = C_LINE; p.line.width = Pt(0.75)
    p.shadow.inherit = False; return p

IMG = pathlib.Path(__file__).parent / "img"

def picture(s, name, x, y, w=None, h=None, frame=True):
    """Place a screenshot, sized by its true aspect ratio, with a hairline frame."""
    from PIL import Image
    f = IMG / name
    iw, ih = Image.open(f).size
    if w is None: w = Emu(int(h * iw / ih))
    if h is None: h = Emu(int(w * ih / iw))
    if frame:
        fr = s.shapes.add_shape(1, x - Emu(9525), y - Emu(9525), w + Emu(19050), h + Emu(19050))
        fr.fill.background(); fr.line.color.rgb = C_LINE; fr.line.width = Pt(1)
        fr.shadow.inherit = False
    return s.shapes.add_picture(str(f), x, y, w, h)

BRAND = pathlib.Path(__file__).parent.parent / "brand"   # ~/APPS/Money2020/brand

def brand_rule(s):
    """Dell blue -> NVIDIA green rule across the top, matching the demo UI."""
    half = Emu(int(W / 2))
    a = s.shapes.add_shape(1, 0, 0, half, Emu(38100))
    a.fill.solid(); a.fill.fore_color.rgb = C_DELL; a.line.fill.background(); a.shadow.inherit = False
    b = s.shapes.add_shape(1, half, 0, half, Emu(38100))
    b.fill.solid(); b.fill.fore_color.rgb = C_GREEN; b.line.fill.background(); b.shadow.inherit = False

def _logo(*names):
    for n in names:
        for ext in ("png","svg"):
            f = BRAND / f"{n}.{ext}"
            if f.exists() and ext == "png":
                return f
    return None

def brand_mark(s, x, y, h=Inches(0.26), small=True):
    """Official logos when present in brand/, otherwise typographic wordmarks."""
    from PIL import Image
    dell = _logo("dell-technologies-white","dell-technologies","dell")
    nv   = _logo("nvidia-white","nvidia")
    cx = x
    if dell:
        iw, ih = Image.open(dell).size
        s.shapes.add_picture(str(dell), cx, y, Emu(int(h*iw/ih)), h); cx += Emu(int(h*iw/ih)) + Inches(0.22)
    else:
        sz = 12 if small else 15
        tf = tb(s, cx, y - Emu(12000), Inches(2.4), Inches(0.3))
        line(tf, "Dell Technologies", sz, C_TEXT, True, SANS, 0, first=True)
        cx += Inches(1.30 if small else 1.68)
    sep = s.shapes.add_shape(1, cx, y - Emu(6000), Emu(9525), h + Emu(12000))
    sep.fill.solid(); sep.fill.fore_color.rgb = C_LINE; sep.line.fill.background(); sep.shadow.inherit = False
    cx += Inches(0.22)
    if nv:
        iw, ih = Image.open(nv).size
        nh = Emu(int(h * 1.62)) if ih > iw * 0.5 else h   # stacked lockup needs more height
        ny = y - Emu(int((nh - h) / 2))                   # optically centre against Dell
        s.shapes.add_picture(str(nv), cx, ny, Emu(int(nh*iw/ih)), nh)
    else:
        tf = tb(s, cx, y - Emu(12000), Inches(1.4), Inches(0.3))
        line(tf, "NVIDIA", 12 if small else 15, C_GREEN, True, SANS, 0, first=True)

def footer(s, page=None):
    brand_mark(s, Inches(0.8), Inches(6.90))
    if page:
        tf = tb(s, Inches(11.0), Inches(6.93), Inches(1.5), Inches(0.3), PP_ALIGN.RIGHT)
        line(tf, page, 10, C_DIM, False, MONO, 0, first=True, align=PP_ALIGN.RIGHT)

def header(s, eyebrow, title):
    tf = tb(s, Inches(0.8), Inches(0.55), Inches(11.7), Inches(0.3))
    line(tf, eyebrow.upper(), 11, C_GREEN, True, MONO, 4, first=True)
    tf2 = tb(s, Inches(0.8), Inches(0.95), Inches(11.7), Inches(0.7))
    line(tf2, title, 30, C_TEXT, True, SANS, 0, first=True)
    rule(s, Inches(0.8), Inches(1.72), Inches(1.5))

def bullets(s, items, x=Inches(0.8), y=Inches(2.15), w=Inches(11.7), size=16, gap=13):
    tf = tb(s, x, y, w, Inches(4.5))
    for i,(head, body) in enumerate(items):
        line(tf, head, size+2, C_TEXT, True, SANS, 3, first=(i==0))
        if body: line(tf, body, size-2, C_DIM, False, SANS, gap)
    return tf

def notes(s, text):
    s.notes_slide.notes_text_frame.text = text

# ---------------------------------------------------------------- 1 title
s = slide()
brand_mark(s, Inches(0.8), Inches(0.75), h=Inches(0.38), small=False)
rule(s, Inches(0.8), Inches(2.35), Inches(2.2))
tf = tb(s, Inches(0.8), Inches(2.6), Inches(11.5), Inches(1.6))
line(tf, "Detecting Fraud in the Graph", 46, C_TEXT, True, SANS, 6, first=True)
line(tf, "Graph neural networks + XGBoost, running entirely on one NVIDIA GB10", 21, C_GREEN, False, SANS, 0)
tf2 = tb(s, Inches(0.8), Inches(4.5), Inches(11.5), Inches(0.9))
line(tf2, "24.4 million card transactions  ·  589,000 scored per second  ·  a regulator-grade "
          "explanation for every decision  ·  no cloud, no network", 15, C_DIM, False, SANS, 0, first=True)
tf3 = tb(s, Inches(0.8), Inches(6.4), Inches(11.5), Inches(0.4))
line(tf3, "MONEY20/20 MIDDLE EAST  ·  RIYADH  ·  14–16 SEPTEMBER 2026", 11, C_DIM, True, MONO, 0, first=True)
notes(s, "Open by putting a hand on the box. The hardware is the story: this entire system runs "
         "on one desktop machine, disconnected.")

# ---------------------------------------------------------------- 1b what is on the box
s = slide(); header(s, "One machine", "Four financial-services demos, all on-device")
rows = [
 ("Fraud detection", C_GREEN,
  "Graph neural network + XGBoost scoring card transactions in real time, with a "
  "Shapley explanation behind every decision.",
  "589,528 txn/sec  ·  F1 0.958"),
 ("Portfolio optimisation", C_DELLL,
  "Mean-CVaR optimisation on NVIDIA cuOpt — robust tail-risk allocation fast enough "
  "to iterate on, not batch overnight.",
  "18x vs CPU  ·  same optimum"),
 ("Document intelligence", C_AMBER,
  "Arabic financial document extraction into tables, text and a knowledge graph — "
  "the KYC and trade-finance back office.",
  "on-device OCR + LLM"),
 ("Loan origination", C_DELL,
  "A 7B vision-language model reading identity and income documents, with a state "
  "machine driving approval and deterministic cross-document fraud checks.",
  "fine-tuned Qwen2.5-VL-7B"),
]
y = Inches(2.0)
for name, col, body, metric in rows:
    panel(s, Inches(0.8), y, Inches(11.7), Inches(1.06))
    bar = s.shapes.add_shape(1, Inches(0.8), y, Emu(38100), Inches(1.06))
    bar.fill.solid(); bar.fill.fore_color.rgb = col; bar.line.fill.background(); bar.shadow.inherit = False
    tf = tb(s, Inches(1.15), y + Inches(0.15), Inches(7.3), Inches(0.85))
    line(tf, name, 15, col, True, SANS, 3, first=True)
    line(tf, body, 11.5, C_DIM, False, SANS, 0)
    tfm = tb(s, Inches(8.6), y + Inches(0.36), Inches(3.7), Inches(0.5), PP_ALIGN.RIGHT)
    line(tfm, metric, 11.5, C_TEXT, True, MONO, 0, first=True, align=PP_ALIGN.RIGHT)
    y += Inches(1.18)
tf = tb(s, Inches(0.8), Inches(6.72), Inches(11.7), Inches(0.4))
line(tf, "No cloud. No network at run time. One NVIDIA GB10 on the table.",
     14, C_GREEN, True, SANS, 0, first=True)
notes(s, "Set the frame: this is a platform claim, not one demo. Then go deep on fraud.")

# ---------------------------------------------------------------- 2 problem
s = slide(); header(s, "The problem", "One fraud in every 819 transactions")
bullets(s, [
 ("A model that says “never fraud” is 99.88% accurate — and worthless.",
  "In the IBM TabFormer dataset: 24,386,900 transactions, 29,757 of them fraudulent. A base rate of 0.122%."),
 ("Extreme imbalance breaks conventional accuracy thinking.",
  "The cost of a miss and the cost of a false alarm are wildly asymmetric, and both are business decisions, not model settings."),
 ("Fraud does not look unusual on its own row.",
  "The fraudulent transactions in this dataset are mostly ordinary amounts at ordinary times. Nothing in the row screams fraud."),
])
p = panel(s, Inches(0.8), Inches(5.3), Inches(11.7), Inches(1.15))
tf = tb(s, Inches(1.15), Inches(5.6), Inches(11.0), Inches(0.7))
line(tf, "So the signal is not in the transaction. It is in the relationships around it.",
     19, C_GREEN, True, SANS, 0, first=True)
notes(s, "Set up the imbalance honestly and early. It is the reason the rest of the approach exists.")

# ---------------------------------------------------------------- 3 evidence
s = slide(); header(s, "What the data says", "The strongest fraud signals are entities, not amounts")
tf = tb(s, Inches(0.8), Inches(2.05), Inches(11.7), Inches(0.4))
line(tf, "Correlation with the fraud label, measured during preprocessing of all 24.4M rows:",
     14, C_DIM, False, SANS, 10, first=True)
rows = [("Merchant state",35.9,True),("Merchant identity",34.9,True),("Merchant city",32.5,True),
        ("Postal code",15.0,False),("Merchant category (MCC)",12.7,False),
        ("Card / cardholder",6.6,False),("Day of month",0.3,False),("Month",0.2,False)]
y = Inches(2.6)
for name, val, hot in rows:
    tfa = tb(s, Inches(0.8), y, Inches(3.0), Inches(0.3))
    line(tfa, name, 14, C_TEXT if hot else C_DIM, hot, SANS, 0, first=True)
    bar = s.shapes.add_shape(1, Inches(4.0), y + Emu(45000), Inches(6.6*val/36.0), Inches(0.20))
    bar.fill.solid(); bar.fill.fore_color.rgb = C_GREEN if hot else C_LINE
    bar.line.fill.background(); bar.shadow.inherit = False
    tfv = tb(s, Inches(10.9), y, Inches(1.2), Inches(0.3))
    line(tfv, f"{val:.1f}%", 13, C_TEXT if hot else C_DIM, hot, MONO, 0, first=True)
    y += Inches(0.46)
tf = tb(s, Inches(0.8), Inches(6.35), Inches(11.7), Inches(0.5))
line(tf, "Where and with whom — not when, and not how much. That is a graph problem.",
     16, C_GREEN, True, SANS, 0, first=True)
notes(s, "These numbers came out of our own preprocessing run, not a paper. They justify the architecture.")

# ---------------------------------------------------------------- 4 approach
s = slide(); header(s, "The approach", "Model the transaction graph, then classify")
cols = [
 ("1  Build the graph", C_GREEN,
  "Cardholders, transactions and merchants become nodes. Every payment becomes edges:\n"
  "user → transaction → merchant.\n\n301,524 training transactions across\n4,873 cardholders and 42,942 merchants."),
 ("2  Learn the neighbourhood", C_GREEN,
  "A GraphSAGE network aggregates each transaction's neighbourhood into an embedding — "
  "so a transaction is represented by the company it keeps, not just its own fields.\n\n"
  "2 hops, 32 hidden channels."),
 ("3  Classify and explain", C_GREEN,
  "XGBoost scores the embeddings, and Shapley value sampling attributes each decision back "
  "to human-readable feature groups.\n\nEvery score arrives with its reasons."),
]
x = Inches(0.8)
for title, col, body in cols:
    panel(s, x, Inches(2.15), Inches(3.68), Inches(3.9))
    tf = tb(s, x+Inches(0.3), Inches(2.45), Inches(3.1), Inches(3.3))
    line(tf, title, 17, col, True, SANS, 10, first=True)
    line(tf, body, 13, C_DIM, False, SANS, 0)
    x += Inches(4.0)
tf = tb(s, Inches(0.8), Inches(6.3), Inches(11.7), Inches(0.5))
line(tf, "Temporal split, not random: train before 2018, validate on 2018, test after. It is a backtest.",
     14, C_TEXT, False, SANS, 0, first=True)
notes(s, "The temporal split matters to anyone who has been burned by leakage. Say it unprompted.")

# ---------------------------------------------------------------- 5 the screen
s = slide(); header(s, "The demo", "What the booth screen shows")
picture(s, "demo-full.png", Inches(2.07), Inches(2.05), w=Inches(9.2))
tf = tb(s, Inches(2.07), Inches(6.98), Inches(9.2), Inches(0.45))
line(tf, "Left: transactions scoring in real time.   Centre: the flagged transaction and its real "
         "details.   Right: why it was flagged.", 13, C_DIM, False, SANS, 0, first=True)
notes(s, "Live output from the running system, captured on the GB10 - not a mockup. "
         "Everything on this slide is reproducible on the stand.")

# ---------------------------------------------------------------- 6 the moment
s = slide(); header(s, "The moment", "An ordinary amount, flagged at 99.5%")
picture(s, "demo-verdict.png", Inches(0.85), Inches(2.05), h=Inches(4.35))
picture(s, "demo-explain.png", Inches(4.65), Inches(2.05), h=Inches(4.35))
tfx = tb(s, Inches(8.9), Inches(2.15), Inches(3.6), Inches(4.2))
line(tfx, "Read the bars, not the numbers.", 17, C_TEXT, True, SANS, 10, first=True)
line(tfx, "Merchant city dominates the decision. Postal code follows. "
          "Transaction amount contributes under 1% as much — a sliver.", 14, C_DIM, False, SANS, 14)
line(tfx, "A rules engine watching for large or unusual amounts never sees this transaction.",
     14, C_TEXT, False, SANS, 14)
line(tfx, "The fraud is not in the payment.\nIt is in the company the payment keeps.",
     15, C_GREEN, True, SANS, 0)
notes(s, "This is the emotional centre of the pitch. Let the bar chart do the talking: "
         "full-width red at the top, a sliver at the bottom for amount.")

# ---------------------------------------------------------------- 6 performance
s = slide(); header(s, "Measured on the box", "Performance on a single NVIDIA GB10")
mets = [("589,528","transactions scored per second","25,803 in 0.044 s"),
        ("0.958","F1 score","precision 94.3% · recall 97.3%"),
        ("41.7 s","to preprocess 24.4M rows","graph construction on GPU"),
        ("23.0 s","to train end to end","GNN + XGBoost"),
        ("3.3 s","per live explanation","Shapley, computed on demand"),
        ("9.4 s","full power-cycle recovery","container restart to serving")]
x, y = Inches(0.8), Inches(2.15)
for i,(big,lab,sub) in enumerate(mets):
    panel(s, x, y, Inches(3.68), Inches(1.75))
    tf = tb(s, x+Inches(0.3), y+Inches(0.22), Inches(3.1), Inches(1.4))
    line(tf, big, 30, C_GREEN, True, MONO, 2, first=True)
    line(tf, lab, 13, C_TEXT, False, SANS, 2)
    line(tf, sub, 11, C_DIM, False, SANS, 0)
    x += Inches(4.0)
    if i == 2: x, y = Inches(0.8), Inches(4.15)
tf = tb(s, Inches(0.8), Inches(6.25), Inches(11.7), Inches(0.6))
line(tf, "Portfolio optimisation: 18x vs CPU on a 51,502-variable LP.   "
         "Document intelligence: full OCR, graph and vector pipeline on the same machine.",
     13.5, C_DIM, False, SANS, 0, first=True)
notes(s, "Every figure here was measured on this machine. None of it is from a datasheet.")

# ---------------------------------------------------------------- 7 explainability
s = slide(); header(s, "Why explainability is the product", "A score a bank cannot justify is a score it cannot use")
bullets(s, [
 ("Every decision carries its reasons.",
  "Shapley attribution runs per transaction, not as a dataset-level feature-importance chart. “Why was this card declined?” has an answer."),
 ("Attribution is grouped into human language.",
  "Not 38 anonymous encoded columns — merchant city, postal code, entry mode, terminal errors, transaction amount."),
 ("It supports the conversation with the regulator and the customer.",
  "Adverse-action explanation, dispute handling, model risk documentation and audit all need the same thing: a defensible reason."),
 ("It also exposes the model's mistakes.",
  "The demo displays ground truth on every transaction, including its 123 false positives and 56 misses. Confidence, not concealment."),
])
notes(s, "In SAMA-regulated institutions this is the difference between a pilot and production.")

# ---------------------------------------------------------------- 7b embedding space
s = slide(); header(s, "Beyond supervised learning", "A foundation model for spending")
picture(s, "embedding-tab.png", Inches(0.8), Inches(2.05), w=Inches(7.4))
tfx = tb(s, Inches(8.5), Inches(2.15), Inches(4.0), Inches(4.3))
line(tfx, "29M parameters. A 6,251-token financial vocabulary. Trained only to predict a "
          "cardholder's next transaction — never shown a fraud label.", 13, C_DIM, False, SANS, 12, first=True)
line(tfx, "Fraud clusters anyway.", 16, C_TEXT, True, SANS, 12)
line(tfx, "Adding these embeddings to the production fraud features lifted average precision "
          "by 26.9% — fewer false alarms at the same catch rate.", 13, C_DIM, False, SANS, 12)
line(tfx, "Embeddings alone score far worse than plain features. They complement feature "
          "engineering; they do not replace it.", 12, C_AMBER, False, SANS, 0)
tf = tb(s, Inches(0.8), Inches(6.6), Inches(11.7), Inches(0.4))
line(tf, "59,123 transactions from a 1.2M corpus  ·  projected with cuML UMAP in 2.2 s on the GB10",
     11.5, C_DIM, False, SANS, 0, first=True)
notes(s, "The GenAI angle. Say plainly that embeddings complement rather than replace - "
         "a data scientist will ask, and the honest answer is the stronger one.")

# ---------------------------------------------------------------- 8 use cases divider
s = slide()
rule(s, Inches(0.8), Inches(3.0), Inches(2.2))
tf = tb(s, Inches(0.8), Inches(3.25), Inches(11.5), Inches(1.4))
line(tf, "Where this applies", 42, C_TEXT, True, SANS, 8, first=True)
line(tf, "The demo is card fraud. The architecture is a pattern for any problem where "
         "the answer lives in relationships.", 18, C_DIM, False, SANS, 0)
notes(s, "Transition: stop selling the demo, start selling the pattern.")

# ---------------------------------------------------------------- 9-12 use cases
UC = [
 ("Use case 01", "Payment fraud at authorisation",
  "Score every card transaction in the authorisation window, with the reason attached.",
  [("What changes","Nothing — this is the demonstrated system."),
   ("The graph","Cardholder → transaction → merchant."),
   ("Why graph wins","Catches ordinary-looking amounts at compromised merchants that amount-and-velocity rules miss."),
   ("Operational fit","Sub-millisecond scoring leaves budget for the rest of the authorisation path.")],
  "Demonstrated today"),
 ("Use case 02", "Money laundering and mule networks",
  "Find the structure, not the suspicious individual: layering chains, mule rings, circular flows.",
  [("What changes","Node types become accounts and counterparties; edges become transfers with amount and timing."),
   ("The graph","Account → transfer → account, across institutions where data sharing permits."),
   ("Why graph wins","Mule networks are invisible per-account and obvious as a subgraph. This is the canonical GNN case."),
   ("Operational fit","Batch scoring over a transfer window; explanations support the STR narrative.")],
  "Strong architectural fit — not yet built"),
 ("Use case 03", "Merchant and acquirer risk",
  "Score the merchant, not just the payment: onboarding risk, transaction laundering, bust-out patterns.",
  [("What changes","Predict on the merchant node instead of the transaction node."),
   ("The graph","Merchant ↔ shared cardholders, terminals, beneficial owners, settlement accounts."),
   ("Why graph wins","Fraudulent merchants are identifiable by who they share customers and infrastructure with."),
   ("Operational fit","Daily or weekly re-scoring; feeds portfolio monitoring and onboarding decisions.")],
  "Same node-prediction pattern"),
 ("Use case 04", "Account takeover and synthetic identity",
  "Detect accounts that behave like a network artefact rather than a person.",
  [("What changes","Add device, session and identity-attribute nodes to the graph."),
   ("The graph","Customer ↔ device ↔ session ↔ shared attributes (address, phone, employer)."),
   ("Why graph wins","Synthetic identities are built from recombined real attributes — the reuse is only visible across accounts."),
   ("Operational fit","Real-time at login and step-up; explanations justify the friction applied.")],
  "Requires additional data sources"),
]
for eyebrow, title, sub, rows, status in UC:
    s = slide(); header(s, eyebrow, title)
    tf = tb(s, Inches(0.8), Inches(2.02), Inches(11.7), Inches(0.4))
    line(tf, sub, 15, C_GREEN, False, SANS, 0, first=True)
    y = Inches(2.72)
    for k, v in rows:
        tfa = tb(s, Inches(0.8), y, Inches(2.7), Inches(0.5))
        line(tfa, k.upper(), 11, C_DIM, True, MONO, 0, first=True)
        tfb = tb(s, Inches(3.7), y-Emu(20000), Inches(8.8), Inches(0.75))
        line(tfb, v, 15, C_TEXT, False, SANS, 0, first=True)
        y += Inches(0.82)
    panel(s, Inches(0.8), Inches(6.15), Inches(11.7), Inches(0.62))
    tfc = tb(s, Inches(1.1), Inches(6.32), Inches(11.0), Inches(0.35))
    line(tfc, f"STATUS   {status}", 12, C_AMBER if "not" in status or "Requires" in status else C_GREEN,
         True, MONO, 0, first=True)
    notes(s, "Be explicit about what is built versus what is architecturally adjacent. "
             "Credibility depends on not blurring the two.")

# ---------------------------------------------------------------- 12b portfolio optimisation
s = slide()
rule(s, Inches(0.8), Inches(3.0), Inches(2.2), C_DELLL)
tf = tb(s, Inches(0.8), Inches(3.25), Inches(11.5), Inches(1.4))
line(tf, "Demo two — Portfolio optimisation", 40, C_TEXT, True, SANS, 8, first=True)
line(tf, "The same box, a different discipline: robust tail-risk allocation on NVIDIA cuOpt.",
     18, C_DIM, False, SANS, 0)
notes(s, "Transition. This widens the conversation from fraud ops to treasury and wealth.")

s = slide(); header(s, "Portfolio optimisation", "Robust allocation, fast enough to iterate")
picture(s, "portfolio-ui.png", Inches(0.8), Inches(2.05), w=Inches(7.3))
tfx = tb(s, Inches(8.4), Inches(2.1), Inches(4.1), Inches(4.4))
line(tfx, "Mean-CVaR asks a harder question than mean-variance: on the worst 5% of days, "
          "how much do I lose? Answering it means simulating tens of thousands of futures.",
     13, C_DIM, False, SANS, 12, first=True)
mets = [("11.7 s","GPU · cuOpt"),("211.7 s","CPU · CVXPY"),("18x","faster, same optimum")]
for big, lab in mets:
    line(tfx, big, 21, C_DELLL, True, MONO, 1)
    line(tfx, lab, 11.5, C_DIM, False, SANS, 10)
line(tfx, "500 assets × 50,000 scenarios — an LP with 51,502 variables.", 12, C_TEXT, False, SANS, 0)
tf = tb(s, Inches(0.8), Inches(6.6), Inches(11.7), Inches(0.45))
line(tf, "Rebalancing is a repeated task, not a one-off. Across hundreds of backtest solves, "
         "that gap turns an overnight job into a morning's work.", 13.5, C_GREEN, True, SANS, 0, first=True)
notes(s, "Lead with the cuOpt solve, never with scenario generation - cuML KDE is not faster "
         "than sklearn at demo scale. And never demo below ~500 assets: at toy sizes the GPU loses.")

# ---------------------------------------------------------------- 12c document intelligence
s = slide()
rule(s, Inches(0.8), Inches(3.0), Inches(2.2), C_AMBER)
tf = tb(s, Inches(0.8), Inches(3.25), Inches(11.5), Inches(1.4))
line(tf, "Demo three — Document intelligence", 40, C_TEXT, True, SANS, 8, first=True)
line(tf, "Arabic financial documents into structured data — on the same box, still offline.",
     18, C_DIM, False, SANS, 0)
notes(s, "The back-office story. Strongest regional differentiator of the three.")

s = slide(); header(s, "Document intelligence", "Arabic documents into tables, text and a graph")
picture(s, "docs-ui.png", Inches(0.8), Inches(2.05), w=Inches(7.3))
tfx = tb(s, Inches(8.4), Inches(2.1), Inches(4.1), Inches(4.4))
line(tfx, "KYC files, trade-finance paperwork and account-opening packs arrive as scans — "
          "often in Arabic, often as tables no rules engine can read.", 13, C_DIM, False, SANS, 12, first=True)
line(tfx, "The pipeline", 12, C_AMBER, True, MONO, 8)
for step in ("Layout detection and OCR on the page",
             "Tables → PostgreSQL, structured and queryable",
             "Text → knowledge graph (Ollama → ArangoDB)",
             "All content → vector store (Qdrant) for retrieval",
             "Source images → object store (MinIO)"):
    line(tfx, "·  " + step, 12.5, C_TEXT, False, SANS, 6)
line(tfx, "Every model runs on the GB10. No document leaves the building.",
     12.5, C_GREEN, True, SANS, 0)
notes(s, "Arabic table extraction is the differentiator - most vendors on that floor will not have it.")

# ---------------------------------------------------------------- 12d loan origination
s = slide()
rule(s, Inches(0.8), Inches(3.0), Inches(2.2), C_DELL)
tf = tb(s, Inches(0.8), Inches(3.25), Inches(11.5), Inches(1.4))
line(tf, "Demo four — Loan origination", 40, C_TEXT, True, SANS, 8, first=True)
line(tf, "A vision-language model reading the documents a loan officer reads, and a state "
         "machine making the decision auditable.", 18, C_DIM, False, SANS, 0)
notes(s, "Retail banking. Pairs naturally with document intelligence - same box, deeper workflow.")

s = slide(); header(s, "Loan origination", "From document upload to an auditable decision")
picture(s, "loan-ui.png", Inches(0.8), Inches(2.05), w=Inches(7.3))
tfx = tb(s, Inches(8.4), Inches(2.1), Inches(4.1), Inches(4.5))
stages = [("01","Capture","Applicant uploads identity and income documents."),
          ("02","Read","A fine-tuned 7B vision model reads each one natively — no OCR-then-parse, no per-template rules."),
          ("03","Cross-check","Deterministic validation across documents: name, employer, declared income."),
          ("04","Decide","A state machine walks the approval path and records why.")]
first = True
for num, title, body in stages:
    line(tfx, f"{num}   {title}", 13.5, C_DELLL, True, MONO, 3, first=first); first = False
    line(tfx, body, 12, C_DIM, False, SANS, 11)
line(tfx, "Fine-tuned on Egyptian documents:\nLoRA r=64, 4 epochs, eval loss 0.139,\non Qwen2.5-VL-7B-Instruct.",
     11.5, C_TEXT, False, MONO, 0)
panel(s, Inches(0.8), Inches(6.45), Inches(11.7), Inches(0.6))
tfc = tb(s, Inches(1.1), Inches(6.6), Inches(11.1), Inches(0.4))
line(tfc, "The cross-document checks are arithmetic, not model output — so the model can be wrong "
          "without the decision being unsafe.", 13, C_GREEN, True, SANS, 0, first=True)
notes(s, "The deterministic cross-document check is the strongest point for a regulated buyer: "
         "the model extracts, but the decision rests on arithmetic that cannot hallucinate. "
         "Adapter status is visible at /health - confirm it says '+ LoRA' before demoing.")

# ---------------------------------------------------------------- 13 sovereign
s = slide(); header(s, "Why it runs on one box", "Data residency without a cloud dependency")
bullets(s, [
 ("The entire system is on-device.",
  "Dataset, preprocessing, training, inference and explanation all run on a single GB10. During the demo the network cable is unplugged and nothing changes."),
 ("No transaction data leaves the institution — or the Kingdom.",
  "That removes the cross-border data-transfer question from the procurement conversation entirely, rather than answering it."),
 ("Deployable where the data already is.",
  "A desktop-class box fits in a branch, a data centre rack, or a regional processing site without redesigning the estate."),
 ("Recovery is measured in seconds.",
  "Full restart to serving in 9.4 seconds, with automatic restart on power loss."),
])
notes(s, "For Saudi institutions this is often the deciding factor, ahead of model quality.")

# ---------------------------------------------------------------- 14 honesty
s = slide(); header(s, "What we are not claiming", "The caveats, stated first")
bullets(s, [
 ("These metrics come from a rebalanced benchmark, not production conditions.",
  "The held-out set runs at 8.09% fraud; the true base rate is 0.122%. At the real rate, false-positive economics change materially. Any pilot must be re-measured on live distributions."),
 ("Training undersamples the majority class.",
  "A 0.1 fraud ratio with computed class weights. Standard practice for extreme imbalance, and a deliberate modelling choice rather than a hidden one."),
 ("The dataset is public and synthetic-adjacent.",
  "IBM TabFormer, Apache 2.0. It demonstrates the method; it is not a proxy for any institution's own fraud patterns."),
 ("A production deployment needs the surrounding system.",
  "Case management, feedback loops, model monitoring, champion/challenger and drift detection are all out of scope for this demo."),
])
notes(s, "Volunteering this early buys more trust than it costs. Risk teams will find it anyway.")

# ---------------------------------------------------------------- 15 next
s = slide(); header(s, "Next steps", "From demo to pilot")
bullets(s, [
 ("Run it on your data.",
  "The pipeline is three commands and roughly a minute of compute for a dataset this size. The graph schema adapts to your transaction model."),
 ("Choose the use case with the clearest data.",
  "Payment fraud and merchant risk need only data an issuer or acquirer already holds. AML and identity graphs need more sourcing."),
 ("Measure at the true base rate from day one.",
  "Agree the false-positive budget with the fraud operations team before tuning anything."),
 ("Deploy where the data lives.",
  "One GB10 per site, or scale up the same containers on larger NVIDIA infrastructure — the stack is identical."),
])
p = panel(s, Inches(0.8), Inches(5.95), Inches(11.7), Inches(0.85))
tf = tb(s, Inches(1.15), Inches(6.18), Inches(11.0), Inches(0.45))
line(tf, "Built on the NVIDIA AI Blueprint for Financial Fraud Detection  ·  "
         "IBM TabFormer dataset (Apache 2.0)  ·  running on NVIDIA GB10",
     13, C_DIM, False, SANS, 0, first=True)
notes(s, "Close on the box again. Offer to run their schema through it.")

# footers: skip slide 1 (lockup already at top) and the section divider
for i, sl in enumerate(prs.slides):
    if i == 0: continue
    footer(sl, f"{i+1:02d} / {len(prs.slides._sldIdLst):02d}")

out = str(pathlib.Path(__file__).parent / "GB10-Demos-for-Events.pptx")
prs.save(out); print("saved:", out, f"({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")
