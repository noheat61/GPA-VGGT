# GPA-VGGT project page

Static site for GitHub Pages: no build step, plain HTML/CSS/JS.

```
index.html            page content
static/css/style.css
static/js/main.js     speed/accuracy explorer data, tabs, gallery, KaTeX
static/img/           figures rendered from paper/figures/*.pdf
.nojekyll             serve files as-is (skip Jekyll)
```

## Preview locally

```bash
python -m http.server 8000   # then open http://localhost:8000
```

Or build a single self-contained file (CSS, JS and images inlined) that opens by double-click:

```bash
python3 bundle.py            # writes preview.html (gitignored, not for deployment)
```

## Deploy

1. Push this directory as the root of a GitHub repository.
2. Settings → Pages → Source: *Deploy from a branch*, Branch: `main`, folder `/ (root)`.

## Before publishing

Search `TODO(` in `index.html`: authors, affiliations, venue, Paper/arXiv/Code links and BibTeX
are placeholders while the paper is anonymous. A link button with `data-todo` renders as
"soon" and is not clickable; drop the attribute once its `href` is real.

## Keeping numbers in sync

Every number comes from `../paper/tab/gen/*.tex`. If a table is regenerated, update both
the HTML tables in `index.html` and the `TIME` / `AUC3` arrays in `static/js/main.js`.
Figures were rendered with PyMuPDF (plots as SVG/PNG, point clouds as JPEG at 220 dpi).
