#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# First, clean previous builds
echo "[AReaL] Cleaning previous builds..."
rm -rf "$SCRIPT_DIR/_build" 2>/dev/null || true

# Create _static directories and copy lang-toggle files for en version
echo "[AReaL] Setting up en static files..."
mkdir -p "$SCRIPT_DIR/en/_static/js" "$SCRIPT_DIR/en/_static/css"
cp "$SCRIPT_DIR/_static/js/lang-toggle.js" "$SCRIPT_DIR/en/_static/js/" 2>/dev/null || true
cp "$SCRIPT_DIR/_static/css/lang-toggle.css" "$SCRIPT_DIR/en/_static/css/" 2>/dev/null || true

# Create _static directories and copy lang-toggle files for zh version
echo "[AReaL] Setting up zh static files..."
mkdir -p "$SCRIPT_DIR/zh/_static/js" "$SCRIPT_DIR/zh/_static/css"
cp "$SCRIPT_DIR/_static/js/lang-toggle.js" "$SCRIPT_DIR/zh/_static/js/" 2>/dev/null || true
cp "$SCRIPT_DIR/_static/css/lang-toggle.css" "$SCRIPT_DIR/zh/_static/css/" 2>/dev/null || true

# Copy figures directory to source directories (so Sphinx can find images within source tree)
echo "[AReaL] Copying figures to source directories..."
cp -r "$SCRIPT_DIR/figures" "$SCRIPT_DIR/en/" 2>/dev/null || true
cp -r "$SCRIPT_DIR/figures" "$SCRIPT_DIR/zh/" 2>/dev/null || true

echo "[AReaL] Building English version..."
uv run --only-dev jupyter-book build "$SCRIPT_DIR/en" --all --path-output "$SCRIPT_DIR/_build/en"

echo "[AReaL] Building Chinese version..."
uv run --only-dev jupyter-book build "$SCRIPT_DIR/zh" --all --path-output "$SCRIPT_DIR/_build/zh"

# Flatten directory structure (move HTML from nested _build/html/ to parent)
echo "[AReaL] Flattening directory structure..."

# Move en HTML files from nested directory to correct location
mkdir -p "$SCRIPT_DIR/_build/en"
if [ -d "$SCRIPT_DIR/_build/en/_build/html" ]; then
  mv "$SCRIPT_DIR/_build/en/_build/html/"* "$SCRIPT_DIR/_build/en/" 2>/dev/null || true
  rmdir "$SCRIPT_DIR/_build/en/_build/html" "$SCRIPT_DIR/_build/en/_build" 2>/dev/null || true
fi

# Move zh HTML files from nested directory to correct location
mkdir -p "$SCRIPT_DIR/_build/zh"
if [ -d "$SCRIPT_DIR/_build/zh/_build/html" ]; then
  mv "$SCRIPT_DIR/_build/zh/_build/html/"* "$SCRIPT_DIR/_build/zh/" 2>/dev/null || true
  rmdir "$SCRIPT_DIR/_build/zh/_build/html" "$SCRIPT_DIR/_build/zh/_build" 2>/dev/null || true
fi

# Copy _static files (lang-toggle) and figures to flattened directories
echo "[AReaL] Copying static files and figures..."

# Copy _static for en (if exists in nested directory)
if [ -d "$SCRIPT_DIR/_build/en/_build/html/_static" ]; then
  cp -r "$SCRIPT_DIR/_build/en/_build/html/_static/"* "$SCRIPT_DIR/_build/en/_static/" 2>/dev/null || true
fi

# Copy figures for en (from source figures directory, which contains all images)
if [ -d "$SCRIPT_DIR/figures" ]; then
  mkdir -p "$SCRIPT_DIR/_build/en/figures"
  cp -r "$SCRIPT_DIR/figures/"* "$SCRIPT_DIR/_build/en/figures/" 2>/dev/null || true
fi

# Copy _static for zh
if [ -d "$SCRIPT_DIR/_build/zh/_build/html/_static" ]; then
  cp -r "$SCRIPT_DIR/_build/zh/_build/html/_static/"* "$SCRIPT_DIR/_build/zh/_static/" 2>/dev/null || true
fi

# Copy figures for zh (from source figures directory)
if [ -d "$SCRIPT_DIR/figures" ]; then
  mkdir -p "$SCRIPT_DIR/_build/zh/figures"
  cp -r "$SCRIPT_DIR/figures/"* "$SCRIPT_DIR/_build/zh/figures/" 2>/dev/null || true
fi

# Clean up temporary _static directories from source tree
echo "[AReaL] Cleaning up temporary static directories..."
rm -rf "$SCRIPT_DIR/en/_static" "$SCRIPT_DIR/zh/_static" 2>/dev/null || true

# Clean up temporary figures directories copied during build
echo "[AReaL] Cleaning up temporary figures directories..."
rm -rf "$SCRIPT_DIR/en/figures" "$SCRIPT_DIR/zh/figures" 2>/dev/null || true

# Also copy top-level js/css directories to _static if they exist (lang-toggle files)
if [ -d "$SCRIPT_DIR/_build/en/js" ]; then
  mkdir -p "$SCRIPT_DIR/_build/en/_static/js"
  cp -r "$SCRIPT_DIR/_build/en/js/"* "$SCRIPT_DIR/_build/en/_static/js/" 2>/dev/null || true
fi
if [ -d "$SCRIPT_DIR/_build/en/css" ]; then
  mkdir -p "$SCRIPT_DIR/_build/en/_static/css"
  cp -r "$SCRIPT_DIR/_build/en/css/"* "$SCRIPT_DIR/_build/en/_static/css/" 2>/dev/null || true
fi

if [ -d "$SCRIPT_DIR/_build/zh/js" ]; then
  mkdir -p "$SCRIPT_DIR/_build/zh/_static/js"
  cp -r "$SCRIPT_DIR/_build/zh/js/"* "$SCRIPT_DIR/_build/zh/_static/js/" 2>/dev/null || true
fi
if [ -d "$SCRIPT_DIR/_build/zh/css" ]; then
  mkdir -p "$SCRIPT_DIR/_build/zh/_static/css"
  cp -r "$SCRIPT_DIR/_build/zh/css/"* "$SCRIPT_DIR/_build/zh/_static/css/" 2>/dev/null || true
fi

# Create root index page with auto-redirect to en (no language selection)
ROOT_INDEX="$SCRIPT_DIR/_build/index.html"
mkdir -p "$SCRIPT_DIR/_build"

cat > "$ROOT_INDEX" <<'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>AReaL Documentation</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body{font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;padding:40px;max-width:720px;margin:auto;color:#222}
  </style>
  <script>
    (function(){
      var stored = null;
      try{stored = localStorage.getItem('areal-doc-lang');}catch(e){console.warn('Could not access localStorage for language preference.', e);}
      var lang = stored === 'zh' ? 'zh' : 'en';
      window.location.replace(lang + '/');
    })();
  </script>
</head>
<body>
  <p>Redirecting to <a href="en/">English</a> / <a href="zh/">中文</a></p>
</body>
</html>
EOF

echo "Build complete!"
echo "  English: _build/en/index.html"
echo "  Chinese: _build/zh/index.html"
echo "  Root:    _build/index.html"
