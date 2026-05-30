UI preview for the Collaborative Robot project

Files:
- index.html — single-page UI that displays the camera and project description
- styles.css — styles for the page

How to open
1) Open the file directly in your browser (Chrome/Edge/Firefox). For camera access some browsers require a secure context (https) or localhost.

2) Serve locally (recommended) from the `ui` directory:

```bash
cd ui
python3 -m http.server 8000
# then open http://localhost:8000 in your browser
```

Notes
- The page will request camera permission. If denied or unavailable it falls back to a background image.
- The background image references `../assets/IMG_6706.jpeg` — replace or edit `styles.css` if you want a different image or a plain color.
