mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << 'TOML'
[server]
headless = true
port = $PORT
enableCORS = false
TOML
