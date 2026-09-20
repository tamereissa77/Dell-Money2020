#!/usr/bin/env python3
import os
import re
import base64
import urllib.parse
from PIL import Image
import io

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

PAGES = [
    {
        'id': 'overview',
        'file': 'index.html',
        'title': 'Dell Technologies · Sovereign & Enterprise AI Showcase',
        'nav_label': 'Overview',
        'nav_sub': 'All six enterprise use cases',
        'accent': 'var(--dell-lt)',
        'category': 'overview'
    },
    {
        'id': 'fraud',
        'file': 'fraud.html',
        'title': 'Dell Technologies · Finding Fraud in the Graph · Enterprise Showcase',
        'nav_label': '01 · Fraud operations',
        'nav_sub': 'Real-time card fraud detection',
        'accent': 'var(--green)',
        'category': 'fin'
    },
    {
        'id': 'portfolio',
        'file': 'portfolio.html',
        'title': 'Dell Technologies · Portfolio Optimisation · Mean-CVaR Tail Risk',
        'nav_label': '02 · Treasury & wealth',
        'nav_sub': 'Portfolio optimisation under tail risk',
        'accent': 'var(--dell-lt)',
        'category': 'fin'
    },
    {
        'id': 'documents',
        'file': 'documents.html',
        'title': 'Dell Technologies · Sovereign Arabic Document Intelligence · Enterprise Showcase',
        'nav_label': '03 · Back office',
        'nav_sub': 'Arabic document intelligence',
        'accent': 'var(--amber)',
        'category': 'fin'
    },
    {
        'id': 'loan',
        'file': 'loan.html',
        'title': 'Dell Technologies · Automated Retail Lending · Enterprise Showcase',
        'nav_label': '04 · Retail lending',
        'nav_sub': 'Automated loan origination',
        'accent': 'var(--dell)',
        'category': 'fin'
    },
    {
        'id': 'vss',
        'file': 'vss.html',
        'title': 'Dell Technologies · Agentic AI for Computer Vision · Video Search & Summarisation',
        'nav_label': '05 · Physical security',
        'nav_sub': 'Video search and summarisation',
        'accent': 'var(--cyan)',
        'category': 'vision'
    },
    {
        'id': 'aiops',
        'file': 'aiops.html',
        'title': 'Dell Technologies · Operational Resilience (AIOps) · Incident Intelligence',
        'nav_label': '06 · Incident intelligence',
        'nav_sub': 'AIOps',
        'accent': 'var(--purple)',
        'category': 'resilience'
    }
]

def image_to_base64(path, max_width=1600, quality=82):
    full_path = os.path.join(BASE_DIR, path.split('?')[0])
    if not os.path.exists(full_path):
        print(f"Warning: image not found {full_path}")
        return path
    
    ext = os.path.splitext(full_path)[1].lower()
    
    if ext in ['.jpg', '.jpeg']:
        try:
            im = Image.open(full_path)
            if im.mode != 'RGB':
                im = im.convert('RGB')
            if im.width > max_width:
                height = int(im.height * (max_width / im.width))
                im = im.resize((max_width, height), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=quality, optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            with open(full_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{b64}"

    elif ext == '.png':
        try:
            im = Image.open(full_path)
            if im.width > max_width:
                height = int(im.height * (max_width / im.width))
                im = im.resize((max_width, height), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format='PNG', optimize=True)
            b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return f"data:image/png;base64,{b64}"
        except Exception as e:
            with open(full_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/png;base64,{b64}"

    elif ext == '.svg':
        with open(full_path, 'r', encoding='utf-8') as f:
            svg_text = f.read()
        encoded = urllib.parse.quote(svg_text)
        return f"data:image/svg+xml;utf8,{encoded}"
    
    else:
        with open(full_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        return f"data:application/octet-stream;base64,{b64}"


def sanitize_money2020(text):
    """Remove any Money20/20 branding, dates, or specific event text."""
    text = re.sub(r'Money20/20\s+Middle\s+East\s*·\s*Riyadh\s*·\s*14–16\s+September\s+2026', 'Dell Technologies & NVIDIA · Enterprise AI Showcase', text, flags=re.I)
    text = re.sub(r'Money20/20\s+Middle\s+East', 'Enterprise AI Showcase', text, flags=re.I)
    text = re.sub(r'Money20/20', 'Enterprise Showcase', text, flags=re.I)
    text = re.sub(r'14–16\s+September\s+2026', 'Executive Briefing Edition', text, flags=re.I)
    text = re.sub(r'Dell Technologies Showcase · [^<]+· Riyadh', 'Dell Technologies & NVIDIA · Sovereign & Enterprise AI Showcase', text, flags=re.I)
    return text


def build():
    print("Building generic Enterprise Offline Showcase (Categorized & Zero Money20/20 branding)...")
    
    # 1. Read and adapt CSS
    with open(os.path.join(BASE_DIR, 'assets', 'style.css'), 'r', encoding='utf-8') as f:
        css = f.read()

    css = sanitize_money2020(css)

    # Inlined SVG pattern
    svg_pattern_b64 = image_to_base64('assets/saudi-pattern.svg')
    css = css.replace("url('saudi-pattern.svg')", f"url('{svg_pattern_b64}')")
    css = css.replace('url("saudi-pattern.svg")', f"url('{svg_pattern_b64}')")
    
    # Additional CSS for Category Headings and SPA Transitions
    extra_css = """
/* Single-file offline SPA view panel styles */
.view-panel {
  display: none;
}
.view-panel.active {
  display: block;
  animation: viewFadeIn .2s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes viewFadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Category Sections on Overview Page */
.category-header {
  margin: 48px 0 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--line);
  display: flex;
  align-items: baseline;
  gap: 16px;
  flex-wrap: wrap;
}
.category-title {
  font-size: 1.3rem;
  font-weight: 650;
  letter-spacing: -0.01em;
  color: #ffffff;
  display: flex;
  align-items: center;
  gap: 10px;
}
.category-title::before {
  content: "";
  display: inline-block;
  width: 10px;
  height: 10px;
  border-radius: 2px;
  background: var(--cat-color, var(--dell-lt));
  box-shadow: 0 0 10px var(--cat-color, var(--dell-lt));
}
.category-desc {
  font-size: 13.5px;
  color: var(--dim);
}
.category-tag {
  margin-left: auto;
  font: 600 10px/1 var(--mono);
  text-transform: uppercase;
  letter-spacing: .12em;
  padding: 4px 10px;
  border-radius: 99px;
  background: rgba(255,255,255,0.06);
  border: 1px solid var(--line);
  color: var(--dimmer);
}
"""
    css += "\n" + extra_css

    # 2. Extract and sanitize main content of all pages
    page_contents = {}

    for p in PAGES:
        filepath = os.path.join(BASE_DIR, p['file'])
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()

        m = re.search(r'<main[^>]*>(.*?)</main>', html, re.DOTALL)
        if m:
            content = m.group(1).strip()
        else:
            content = html.strip()

        # Sanitize any Money20/20 mentions
        content = sanitize_money2020(content)

        # Replace inter-page links with hash links
        for target_page in PAGES:
            content = re.sub(r'href=[\'"]' + re.escape(target_page['file']) + r'[\'"]', f'href="#{target_page["id"]}"', content)
            content = re.sub(r'href=[\'"]/' + re.escape(target_page['file']) + r'[\'"]', f'href="#{target_page["id"]}"', content)

        page_contents[p['id']] = content

    # 3. Restructure Overview Page content to organize use cases into categories
    overview_html = page_contents['overview']

    overview_html = re.sub(
        r'<div class="eyebrow">.*?</div>',
        '<div class="eyebrow">Dell Technologies &amp; NVIDIA · Sovereign &amp; Enterprise AI Showcase</div>',
        overview_html
    )
    overview_html = re.sub(
        r'<h1>AI use cases for financial services</h1>',
        '<h1>Enterprise AI use cases across critical industries</h1>',
        overview_html
    )
    overview_html = re.sub(
        r'<p class="lede">Six problems banks actually have — card fraud, tail risk, Arabic paperwork,\s*loan origination, branch video and operational incidents',
        '<p class="lede">Six mission-critical enterprise challenges — card fraud, portfolio tail risk, Arabic document intelligence, retail lending, branch video intelligence, and operational incidents',
        overview_html
    )
    overview_html = overview_html.replace('assets/img/hero-index.jpg', 'assets/img/hero-enterprise.jpg')

    # Extract individual cards 1 to 6 from overview
    card_pattern = r'(<a\s+class="uc"\s+href="#([a-z]+)".*?</a>)'
    cards_found = re.findall(card_pattern, overview_html, re.DOTALL)
    card_dict = {cf[1]: cf[0] for cf in cards_found}

    categorized_cards_html = f"""
  <!-- ── CATEGORY 1: Financial Services ── -->
  <div class="category-header">
    <div class="category-title" style="--cat-color:var(--green)">Financial Services</div>
    <div class="category-desc">Mission-critical banking, real-time fraud defense, quantitative treasury, and sovereign document automation</div>
    <div class="category-tag">4 Solutions · Sovereign On-Device</div>
  </div>
  <div class="cards">
    {card_dict.get('fraud', '')}
    {card_dict.get('portfolio', '')}
    {card_dict.get('documents', '')}
    {card_dict.get('loan', '')}
  </div>

  <!-- ── CATEGORY 2: Agentic AI for Computer Vision ── -->
  <div class="category-header">
    <div class="category-title" style="--cat-color:var(--cyan)">Agentic AI for Computer Vision</div>
    <div class="category-desc">Vision-Language models (VLM), natural-language temporal video search, and autonomous branch analytics</div>
    <div class="category-tag">Physical Security &amp; Safety</div>
  </div>
  <div class="cards">
    {card_dict.get('vss', '')}
  </div>

  <!-- ── CATEGORY 3: Operational Resilience ── -->
  <div class="category-header">
    <div class="category-title" style="--cat-color:var(--purple)">Operational Resilience</div>
    <div class="category-desc">Autonomous AIOps, cross-tier telemetry topology correlation, and real-time incident remediation</div>
    <div class="category-tag">Incident Intelligence &amp; SRE</div>
  </div>
  <div class="cards">
    {card_dict.get('aiops', '')}
  </div>
"""

    # Precisely replace the section from '<div class="section-note">The use cases</div>' to '<div class="section-note">The machine</div>'
    start_marker = '<div class="section-note">The use cases</div>'
    end_marker = '<div class="section-note">The machine</div>'

    start_idx = overview_html.find(start_marker)
    end_idx = overview_html.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        overview_html = overview_html[:start_idx] + categorized_cards_html.strip() + "\n\n  " + overview_html[end_idx:]
    else:
        print("Warning: Could not find exact start and end markers in overview_html!")

    page_contents['overview'] = overview_html

    # 4. Cache and embed all images in Base64
    image_cache = {}
    
    dell_logo_b64 = image_to_base64('assets/dell-logo.png')
    nvidia_logo_b64 = image_to_base64('assets/nvidia-white.png')

    def replace_img_src(match):
        src = match.group(2)
        if src.startswith('data:') or src.startswith('http://') or src.startswith('https://'):
            return match.group(0)
        
        clean_src = src.lstrip('./').split('?')[0]
        if clean_src not in image_cache:
            print(f"Embedding image: {clean_src}")
            image_cache[clean_src] = image_to_base64(clean_src)
        
        return f'{match.group(1)}="{image_cache[clean_src]}"'

    for pid in page_contents:
        page_contents[pid] = re.sub(r'(src)=[\'"]([^\'"]+)[\'"]', replace_img_src, page_contents[pid])

    # 5. Build Categorized Sidebar Navigation
    sidebar_html = f"""
<aside class="side">
  <div class="lockup">
    <img src="{dell_logo_b64}" alt="Dell Technologies" class="d">
    <span class="sep"></span>
    <img src="{nvidia_logo_b64}" alt="NVIDIA" class="n">
  </div>

  <div class="side-label">Start here</div>
  <nav class="side-nav">
    <a href="#overview" data-view="overview" class="on" style="--ac:var(--dell-lt)">
      <span class="k">Overview</span><span class="t">All six enterprise use cases</span>
    </a>
  </nav>

  <div class="side-label">Financial Services</div>
  <nav class="side-nav">
    <a href="#fraud" data-view="fraud" style="--ac:var(--green)"><span class="k">01 · Fraud operations</span><span class="t">Real-time card fraud detection</span></a>
    <a href="#portfolio" data-view="portfolio" style="--ac:var(--dell-lt)"><span class="k">02 · Treasury &amp; wealth</span><span class="t">Portfolio optimisation under tail risk</span></a>
    <a href="#documents" data-view="documents" style="--ac:var(--amber)"><span class="k">03 · Back office</span><span class="t">Arabic document intelligence</span></a>
    <a href="#loan" data-view="loan" style="--ac:var(--dell)"><span class="k">04 · Retail lending</span><span class="t">Automated loan origination</span></a>
  </nav>

  <div class="side-label">Agentic AI for Computer Vision</div>
  <nav class="side-nav">
    <a href="#vss" data-view="vss" style="--ac:var(--cyan)"><span class="k">05 · Physical security</span><span class="t">Video search and summarisation</span></a>
  </nav>

  <div class="side-label">Operational Resilience</div>
  <nav class="side-nav">
    <a href="#aiops" data-view="aiops" style="--ac:var(--purple)"><span class="k">06 · Incident intelligence</span><span class="t">AIOps</span></a>
  </nav>

  <div class="side-foot">
    <strong>Dell Technologies</strong><br>
    Sovereign &amp; Enterprise AI Solutions<br><br>
    Powered by NVIDIA Grace Blackwell GB10<br>
    <span style="display:inline-block;margin-top:6px;font-size:10px;padding:2px 7px;border-radius:4px;background:rgba(0,118,206,0.15);border:1px solid rgba(0,118,206,0.3);color:var(--dell-lt)">Executive Standalone Edition</span>
  </div>
</aside>
"""

    # 6. Build views HTML cleanly wrapped in view panels
    views_html = []
    for p in PAGES:
        active_class = " active" if p['id'] == 'overview' else ""
        views_html.append(f"""
    <!-- ── VIEW: {p['id'].upper()} ({p['category'].upper()}) ── -->
    <div id="view-{p['id']}" class="view-panel{active_class}" data-title="{p['title']}">
{page_contents[p['id']]}
    </div>
""")

    views_combined = "\n".join(views_html)

    # 7. Generic Footer HTML
    footer_html = """
<footer class="site">
  <div>
    <span>Dell Technologies &amp; NVIDIA · Sovereign &amp; Enterprise AI Solutions</span>
    <span style="color:var(--dimmer)">·</span>
    <span>Autonomous &amp; Air-Gapped Intelligence</span>
    <span class="tag">Production Validated</span>
  </div>
</footer>
"""

    # 8. JavaScript router
    router_js = """
<script>
(function() {
  var titleMap = {
    'overview': 'Dell Technologies · Sovereign & Enterprise AI Showcase',
    'fraud': 'Dell Technologies · Finding Fraud in the Graph · Enterprise Showcase',
    'portfolio': 'Dell Technologies · Portfolio Optimisation · Mean-CVaR Tail Risk',
    'documents': 'Dell Technologies · Sovereign Arabic Document Intelligence · Enterprise Showcase',
    'loan': 'Dell Technologies · Automated Retail Lending · Enterprise Showcase',
    'vss': 'Dell Technologies · Agentic AI for Computer Vision · Video Search & Summarisation',
    'aiops': 'Dell Technologies · Operational Resilience (AIOps) · Incident Intelligence'
  };

  function switchView(viewId) {
    if (!viewId || !titleMap[viewId]) {
      viewId = 'overview';
    }

    // Toggle panels
    var panels = document.querySelectorAll('.view-panel');
    panels.forEach(function(p) {
      if (p.id === 'view-' + viewId) {
        p.classList.add('active');
      } else {
        p.classList.remove('active');
      }
    });

    // Update nav links
    var navLinks = document.querySelectorAll('.side-nav a');
    navLinks.forEach(function(a) {
      if (a.getAttribute('data-view') === viewId) {
        a.classList.add('on');
      } else {
        a.classList.remove('on');
      }
    });

    // Update document title
    if (titleMap[viewId]) {
      document.title = titleMap[viewId];
    }

    // Scroll to top of window
    window.scrollTo(0, 0);
  }

  window.addEventListener('hashchange', function() {
    var target = window.location.hash.replace(/^#/, '');
    switchView(target);
  });

  document.addEventListener('DOMContentLoaded', function() {
    var initial = window.location.hash.replace(/^#/, '');
    switchView(initial || 'overview');

    // Intercept clicks on links with href="#..."
    document.body.addEventListener('click', function(e) {
      var a = e.target.closest('a');
      if (a && a.getAttribute('href') && a.getAttribute('href').startsWith('#')) {
        var id = a.getAttribute('href').replace(/^#/, '');
        if (titleMap[id]) {
          e.preventDefault();
          window.location.hash = id;
          switchView(id);
        }
      }
    });
  });
})();
</script>
"""

    # 9. Assemble Full Standalone HTML
    standalone_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dell Technologies · Sovereign &amp; Enterprise AI Showcase</title>
<style>
{css}
</style>
</head>
<body>
<div class="shell">
{sidebar_html}
  <div class="content">
    <main>
{views_combined}
    </main>
{footer_html}
  </div>
</div>
{router_js}
</body>
</html>
"""

    output_filename = 'dell-enterprise-ai-showcase-offline.html'
    output_path = os.path.join(BASE_DIR, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(standalone_html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Successfully generated generic Enterprise Offline Showcase: {output_filename} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    build()
