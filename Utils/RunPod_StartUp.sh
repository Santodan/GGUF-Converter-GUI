#!/bin/bash
set -euo pipefail

echo "[1/7] Cleanup old sessions..."
pkill -f websockify || true
pkill -f "Xtigervnc|Xvnc|vncserver" || true
vncserver -kill :1 >/dev/null 2>&1 || true
rm -rf /tmp/.X1-lock /tmp/.X11-unix/X1 || true

echo "[2/7] Install system packages (includes cmake + build-essential)..."
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential cmake git curl wget jq \
  python3 python3-pip python3-tk \
  xfce4 xfce4-goodies dbus-x11 xterm autocutsel \
  tightvncserver novnc websockify python3-websockify

echo "[3/7] Install python packages..."
python3 -m pip install --upgrade pip
python3 -m pip install \
  safetensors huggingface_hub tqdm sentencepiece numpy gguf prompt_toolkit requests torch

echo "[4/7] Download app..."
mkdir -p /workspace
cd /workspace
wget -N https://raw.githubusercontent.com/Santodan/GGUF-Converter-GUI/refs/heads/main/gui_run_conversion.py

echo "[5/7] Configure VNC startup (XFCE desktop)..."
mkdir -p /root/.vnc
cat > /root/.vnc/xstartup <<'EOF'
#!/bin/sh
unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS

if command -v dbus-launch >/dev/null 2>&1; then
  eval "$(dbus-launch --sh-syntax)"
fi

autocutsel -fork

# Start a full desktop session
startxfce4 &
EOF
chmod +x /root/.vnc/xstartup

echo "[6/7] Start VNC + noVNC..."
export USER=root

# Set a VNC password (change 'runpod' if you want). noVNC will ask for this.
mkdir -p /root/.vnc
printf "runpod\nrunpod\nn\n" | vncpasswd >/dev/null 2>&1 || true

vncserver :1 -geometry 1280x800 -depth 24

# noVNC on http://<runpod-proxy>/vnc.html
websockify --web /usr/share/novnc 6080 localhost:5901 >/tmp/websockify.log 2>&1 &

echo "[7/7] Start your tkinter GUI on DISPLAY=:1..."
export DISPLAY=:1

# Start GUI, write logs to a file
python3 /workspace/gui_run_conversion.py > /tmp/gguf_gui.log 2>&1 &
GUI_PID=$!

echo ""
echo "=== READY ==="
echo "GUI PID: $GUI_PID"
echo "Open noVNC at: /vnc.html (port 6080 HTTP service)"
echo "Terminal is now dedicated to GUI logs."
echo ""

# Replace the current shell with a log follower (no prompt)
exec tail -n 200 -f /tmp/gguf_gui.log