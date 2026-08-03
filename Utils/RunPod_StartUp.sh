#!/bin/bash

set -euo pipefail

# Directory containing this startup script.
# llama-quantize may be created here by the GUI/build process.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Make executables and shared libraries beside this script discoverable.
export PATH="${SCRIPT_DIR}:${PATH}"
export LD_LIBRARY_PATH="${SCRIPT_DIR}:${LD_LIBRARY_PATH:-}"

# Default mode performs all installation steps.
# --restart skips apt and pip installation.
RESTART_ONLY=false

case "${1:-}" in
    --restart|--skip-install)
        RESTART_ONLY=true
        ;;
    "")
        ;;
    *)
        echo "Usage: $0 [--restart|--skip-install]"
        exit 1
        ;;
esac

echo "[1/7] Cleanup old sessions..."

# Stop an already-running converter GUI.
pkill -f '[g]ui_run_conversion.py' || true

pkill -f '[w]ebsockify' || true
pkill -f 'Xtigervnc|Xvnc|vncserver' || true

vncserver -kill :1 >/dev/null 2>&1 || true

rm -rf /tmp/.X1-lock /tmp/.X11-unix/X1 || true

if [ "$RESTART_ONLY" = false ]; then
    echo "[2/7] Install system packages..."

    apt-get update

    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential cmake git curl wget jq \
        python3 python3-pip python3-tk \
        xfce4 xfce4-goodies dbus-x11 xterm autocutsel \
        tightvncserver novnc websockify python3-websockify

    echo "[3/7] Install Python packages..."

    python3 -m pip install --upgrade pip

    python3 -m pip install \
        safetensors \
        huggingface_hub \
        tqdm \
        sentencepiece \
        numpy \
        gguf \
        prompt_toolkit \
        requests \
        torch
else
    echo "[2/7] Skipping system package installation."
    echo "[3/7] Skipping Python package installation."
fi

echo "[4/7] Update application..."

mkdir -p "$SCRIPT_DIR"
cd "$SCRIPT_DIR"

wget -N \
    https://raw.githubusercontent.com/Santodan/GGUF-Converter-GUI/refs/heads/main/gui_run_conversion.py

# Ensure llama-quantize is executable when it exists.
if [ -f "$SCRIPT_DIR/llama-quantize" ]; then
    chmod +x "$SCRIPT_DIR/llama-quantize"

    echo "Found llama-quantize:"
    command -v llama-quantize
else
    echo "llama-quantize is not present yet."
    echo "It can be built or created by the GUI later."
fi

echo "[5/7] Configure VNC startup..."

mkdir -p /root/.vnc

cat > /root/.vnc/xstartup <<'EOF'
#!/bin/sh

unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS

if command -v dbus-launch >/dev/null 2>&1; then
    eval "$(dbus-launch --sh-syntax)"
fi

autocutsel -fork
startxfce4 &
EOF

chmod +x /root/.vnc/xstartup

echo "[6/7] Start VNC and noVNC..."

export USER=root

mkdir -p /root/.vnc

printf "runpod\nrunpod\nn\n" |
    vncpasswd >/dev/null 2>&1 || true

vncserver :1 -geometry 1280x800 -depth 24

websockify \
    --web /usr/share/novnc \
    6080 \
    localhost:5901 \
    >/tmp/websockify.log 2>&1 &

echo "[7/7] Start converter GUI..."

export DISPLAY=:1

python3 "$SCRIPT_DIR/gui_run_conversion.py" \
    >/tmp/gguf_gui.log 2>&1 &

GUI_PID=$!

echo
echo "=== READY ==="
echo "GUI PID: $GUI_PID"
echo "Application directory: $SCRIPT_DIR"
echo "llama-quantize: $(command -v llama-quantize 2>/dev/null || echo 'not created yet')"
echo "Open noVNC at: /vnc.html (port 6080)"
echo "Following GUI log..."
echo

exec tail -n 200 -f /tmp/gguf_gui.log
