# Inline CSS, JS, images and point clouds of the project page into one self-contained
# HTML file (preview.html) that opens by double-click, no server needed.
import base64, glob, json, mimetypes, re, sys, os
here = os.path.dirname(os.path.abspath(__file__))
root = sys.argv[1] if len(sys.argv) > 1 else here
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "preview.html")
html = open(os.path.join(root, 'index.html')).read()
mimetypes.add_type('image/svg+xml', '.svg')
def b64(p):
    return base64.b64encode(open(p, 'rb').read()).decode()
def data_uri(rel):
    p = os.path.join(root, rel)
    return f"data:{mimetypes.guess_type(p)[0]};base64," + b64(p)
css = open(os.path.join(root, 'static/css/style.css')).read()
js = open(os.path.join(root, 'static/js/main.js')).read()
viewer = open(os.path.join(root, 'static/js/viewer.js')).read()
html = html.replace('<link rel="stylesheet" href="static/css/style.css">', '<style>\n' + css + '</style>')
# Inline script must run after KaTeX loads, so defer it via the load event.
html = html.replace('<script defer src="static/js/main.js"></script>',
                    '<script>\nwindow.addEventListener("load", () => {\n' + js + '\n});\n</script>')
# viewer.js reads window.__CLOUDS__ before falling back to fetch(), which file:// blocks.
cdir = os.path.join(root, 'static/clouds')
clouds = {os.path.relpath(p, cdir): b64(p) for p in sorted(glob.glob(os.path.join(cdir, '**/*'), recursive=True))
          if os.path.isfile(p)}
html = html.replace('<script type="module" src="static/js/viewer.js"></script>',
                    '<script>window.__CLOUDS__ = ' + json.dumps(clouds) + ';</script>\n'
                    '<script type="module">\n' + viewer + '\n</script>')
# The clouds are inlined, so the deployed page's preload hints for them are dropped.
html = re.sub(r'\s*<link rel="preload" href="static/[^"]+"[^>]*>', '', html)
html = re.sub(r'(src|content)="(static/img/[^"]+)"', lambda m: f'{m.group(1)}="{data_uri(m.group(2))}"', html)
left = [s for s in re.findall(r'["\']static/[^"\']*["\']', html) if s.strip('"\'') != 'static/clouds/']
assert not left, left[:3]
open(out, 'w').write(html)
print(out, f"{os.path.getsize(out)/1e6:.1f} MB")
