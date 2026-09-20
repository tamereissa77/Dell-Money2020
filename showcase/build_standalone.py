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
        'title': 'Dell Technologies · AI for Financial Services · Money20/20 Middle East',
        'nav_label': 'Overview',
        'nav_sub': 'All six use cases',
        'accent': 'var(--green)'
    },
    {
        'id': 'fraud',
        'file': 'fraud.html',
        'title': 'Dell Technologies · Finding Fraud in the Graph · Money20/20 Middle East',
        'nav_label': '01 · Fraud operations',
        'nav_sub': 'Real-time card fraud detection',
        'accent': 'var(--green)'
    },
    {
        'id': 'portfolio',
        'file': 'portfolio.html',
        'title': 'Dell Technologies · Portfolio Optimisation · Mean-CVaR Tail Risk',
        'nav_label': '02 · Treasury & wealth',
        'nav_sub': 'Portfolio optimisation under tail risk',
        'accent': 'var(--dell-lt)'
    },
    {
        'id': 'documents',
        'file': 'documents.html',
        'title': 'Dell Technologies · Sovereign Arabic Document Intelligence · Money20/20 Middle East',
        'nav_label': '03 · Back office',
        'nav_sub': 'Arabic document intelligence',
        'accent': 'var(--amber)'
    },
    {
        'id': 'loan',
        'file': 'loan.html',
        'title': 'Dell Technologies · Automated Retail Lending · Sovereign Decisioning',
        'nav_label': '04 · Retail lending',
        'nav_sub': 'Automated loan origination',
        'accent': 'var(--dell)'
    },
    {
        'id': 'vss',
        'file': 'vss.html',
        'title': 'Dell Technologies · Branch Video Intelligence · Sovereign Vision-Language AI',
        'nav_label': '05 · Physical security',
        'nav_sub': 'Video search and summarisation',
        'accent': 'var(--cyan)'
    },
    {
        'id': 'aiops',
        'file': 'aiops.html',
        'title': 'Dell Technologies · Sovereign Incident Intelligence (AIOps) · Money20/20 Middle East',
        'nav_label': '06 · Operational resilience',
        'nav_sub': 'AIOps',
        'accent': 'var(--purple)'
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
            print(f"Error compressing JPEG {full_path}: {e}")
            with open(full_path, 'rb') as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{b64}"

    elif ext == '.png':
        try:
            im = Image.open(full_path)
            # If large photographic PNG, optimize
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


def build():
    print("Building standalone offline HTML file...")
    
    # 1. Read CSS
    with open(os.path.join(BASE_DIR, 'assets', 'style.css'), 'r', encoding='utf-8') as f:
        css = f.read()

    # Replace saudi-pattern.svg in CSS with inline SVG data URI
    svg_pattern_b64 = image_to_base64('assets/saudi-pattern.svg')
    css = css.replace("url('saudi-pattern.svg')", f"url('{svg_pattern_b64}')")
    css = css.replace('url("saudi-pattern.svg")', f"url('{svg_pattern_b64}')")
    
    # Extra CSS for SPA view-panel management
    spa_css = """
/* Single-file offline SPA view panel styles */
.view-panel {
  display: none;
}
.view-panel.active {
  display: block;
  animation: viewFadeIn .22s cubic-bezier(0.16, 1, 0.3, 1);
}
@keyframes viewFadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
"""
    css += "\n" + spa_css

    # 2. Extract main content of all pages
    page_contents = {}
    footer_html = ""

    for p in PAGES:
        filepath = os.path.join(BASE_DIR, p['file'])
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()

        # Extract <main>...</main>
        m = re.search(r'<main[^>]*>(.*?)</main>', html, re.DOTALL)
        if m:
            content = m.group(1)
        else:
            content = html

        # Extract footer if not already found
        if not footer_html:
            fm = re.search(r'(<footer\s+class="site"[^>]*>.*?</footer>)', html, re.DOTALL)
            if fm:
                footer_html = fm.group(1)

        # Replace inter-page links with hash links
        for target_page in PAGES:
            # Replace href="page.html" with href="#page_id"
            content = re.sub(r'href=[\'"]' + re.escape(target_page['file']) + r'[\'"]', f'href="#{target_page["id"]}"', content)
            content = re.sub(r'href=[\'"]/' + re.escape(target_page['file']) + r'[\'"]', f'href="#{target_page["id"]}"', content)

        page_contents[p['id']] = content

    # 3. Cache and replace all image references across all page contents, sidebar, and footer
    # Find all images referenced
    image_cache = {}
    
    # Logos
    dell_logo_b64 = image_to_base64('assets/dell-logo.png')
    nvidia_logo_b64 = image_to_base64('assets/nvidia-white.png')

    # Collect all image sources in contents
    def replace_img_src(match):
        src = match.group(2)
        if src.startswith('data:') or src.startswith('http://') or src.startswith('https://'):
            return match.group(0)
        
        # Clean relative path
        clean_src = src.lstrip('./').split('?')[0]
        if clean_src not in image_cache:
            print(f"Embedding image: {clean_src}")
            image_cache[clean_src] = image_to_base64(clean_src)
        
        return f'{match.group(1)}="{image_cache[clean_src]}"'

    for pid in page_contents:
        page_contents[pid] = re.sub(r'(src)=[\'"]([^\'"]+)[\'"]', replace_img_src, page_contents[pid])

    if footer_html:
        footer_html = re.sub(r'(src)=[\'"]([^\'"]+)[\'"]', replace_img_src, footer_html)

    # 4. Construct Sidebar
    sidebar_html = f"""
<aside class="side">
  <div class="lockup">
    <img src="{dell_logo_b64}" alt="Dell Technologies" class="d">
    <span class="sep"></span>
    <img src="{nvidia_logo_b64}" alt="NVIDIA" class="n">
  </div>
  <div class="side-label">Start here</div>
  <nav class="side-nav">
    <a href="#overview" data-view="overview" class="on" style="--ac:var(--green)">
      <span class="k">Overview</span><span class="t">All six use cases</span>
    </a>
  </nav>
  <div class="side-label">The use cases</div>
  <nav class="side-nav">
    <a href="#fraud" data-view="fraud" style="--ac:var(--green)"><span class="k">01 · Fraud operations</span><span class="t">Real-time card fraud detection</span></a>
    <a href="#portfolio" data-view="portfolio" style="--ac:var(--dell-lt)"><span class="k">02 · Treasury &amp; wealth</span><span class="t">Portfolio optimisation under tail risk</span></a>
    <a href="#documents" data-view="documents" style="--ac:var(--amber)"><span class="k">03 · Back office</span><span class="t">Arabic document intelligence</span></a>
    <a href="#loan" data-view="loan" style="--ac:var(--dell)"><span class="k">04 · Retail lending</span><span class="t">Automated loan origination</span></a>
    <a href="#vss" data-view="vss" style="--ac:var(--cyan)"><span class="k">05 · Physical security</span><span class="t">Video search and summarisation</span></a>
    <a href="#aiops" data-view="aiops" style="--ac:var(--purple)"><span class="k">06 · Operational resilience</span><span class="t">AIOps</span></a>
  </nav>
  <div class="side-foot">
    <strong>Money20/20 Middle East</strong><br>
    Riyadh · 14–16 September 2026<br><br>
    Dell Technologies with NVIDIA<br>
    <span style="display:inline-block;margin-top:6px;font-size:10px;padding:2px 7px;border-radius:4px;background:rgba(0,118,206,0.15);border:1px solid rgba(0,118,206,0.3);color:var(--dell-lt)">Offline Self-Contained Edition</span>
  </div>
</aside>
"""

    # 5. Build views HTML
    views_html = []
    for p in PAGES:
        active_class = " active" if p['id'] == 'overview' else ""
        views_html.append(f"""
    <!-- ── VIEW: {p['id'].upper()} ── -->
    <div id="view-{p['id']}" class="view-panel{active_class}" data-title="{p['title']}">
{page_contents[p['id']]}
    </div>
""")

    views_combined = "\n".join(views_html)

    # 6. JavaScript router
    router_js = """
<script>
(function() {
  var titleMap = {
    'overview': 'Dell Technologies · AI for Financial Services · Money20/20 Middle East',
    'fraud': 'Dell Technologies · Finding Fraud in the Graph · Money20/20 Middle East',
    'portfolio': 'Dell Technologies · Portfolio Optimisation · Mean-CVaR Tail Risk',
    'documents': 'Dell Technologies · Sovereign Arabic Document Intelligence · Money20/20 Middle East',
    'loan': 'Dell Technologies · Automated Retail Lending · Sovereign Decisioning',
    'vss': 'Dell Technologies · Branch Video Intelligence · Sovereign Vision-Language AI',
    'aiops': 'Dell Technologies · Sovereign Incident Intelligence (AIOps) · Money20/20 Middle East'
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

    // Smooth scroll to top of content
    window.scrollTo({ top: 0, behavior: 'instant' });
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

    # 7. Assemble Full Standalone HTML
    standalone_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dell Technologies · AI for Financial Services · Money20/20 Middle East</title>
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

    output_filename = 'money2020-dell-showcase-offline.html'
    output_path = os.path.join(BASE_DIR, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(standalone_html)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Successfully generated standalone offline showcase: {output_filename} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    build()
