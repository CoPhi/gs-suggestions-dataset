// Configurazione runtime del frontend (sovrascrivibile via volume Docker senza rebuild)
window.__env = window.__env || {};
window.__env.apiUrl = 'http://127.0.0.1:8000';
