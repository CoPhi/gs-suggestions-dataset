#!/bin/sh
set -eu

# Genera assets/config.js all'avvio del container a partire da GS_API_URL (es. da Portainer/.env)
API_URL="${GS_API_URL:-http://127.0.0.1:8000}"
# Escapa eventuali backslash e apici singoli per mantenere valido il literal JavaScript
ESCAPED_API_URL=$(printf '%s' "$API_URL" | sed "s/\\\\/\\\\\\\\/g; s/'/\\\\'/g")

mkdir -p /usr/share/nginx/html/assets
cat <<EOF > /usr/share/nginx/html/assets/config.js
// Configurazione runtime generata all'avvio del container da GS_API_URL
window.__env = window.__env || {};
window.__env.apiUrl = '${ESCAPED_API_URL}';
EOF
