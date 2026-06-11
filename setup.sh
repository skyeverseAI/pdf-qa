mkdir -p ~/.streamlit
echo "[server]
headless = true
port = ${PORT:-8501}
enableCORS = false
" > ~/.streamlit/config.toml
