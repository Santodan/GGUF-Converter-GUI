# How to runt he script in RunPod

Before starting, you will need two additional things:
- NGrok token, which is free and you can get it from https://ngrok.com/
- A way to connect to the VNC Server, I'm familiar with MobaXterm, so that's what I'm doing

## 1. Install Desktop & Network Tools
`apt-get update && apt-get install -y python3-tk xauth xfce4 xfce4-goodies tightvncserver xterm dbus-x11 curl git build-essential cmake libcurl4-openssl-dev python3-pip autocutsel`

## 2. Install Ngrok
```
curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list
apt-get update && apt-get install -y ngrok
```

## 3. Download Files
```
cd /workspace
wget -N https://raw.githubusercontent.com/Santodan/GGUF-Converter-GUI/refs/heads/main/gui_run_conversion.py
```

## 4. CLEANUP (Prevents "Address already in use" errors)
```
pkill -f ngrok
rm -rf /tmp/.X1-lock /tmp/.X11-unix/X1
```

## 5. SETUP VNC CONFIG
```
mkdir -p /root/.vnc
echo '#!/bin/sh' > /root/.vnc/xstartup
echo 'unset SESSION_MANAGER' >> /root/.vnc/xstartup
echo 'unset DBUS_SESSION_BUS_ADDRESS' >> /root/.vnc/xstartup
echo 'autocutsel -fork &' >> /root/.vnc/xstartup
echo 'xfwm4 &' >> /root/.vnc/xstartup
echo 'xterm -geometry 80x24+10+10 -ls -title "Rescue Terminal" &' >> /root/.vnc/xstartup
chmod +x /root/.vnc/xstartup
```

## 6. INSTALL PYTHON DEPS
`pip install safetensors huggingface_hub tqdm sentencepiece numpy gguf prompt_toolkit requests`

## 7. AUTH NGROK
`ngrok config add-authtoken <your ngrok token>`

## 8. START SERVICES
```
export USER=root
vncserver :1 -geometry 1280x800 -depth 24
nohup ngrok tcp 5901 > ngrok_debug.log 2>&1 &
```

## 9. GET & PRINT URL
```
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o "tcp://[^\"']*")
$NGROK_URL
```
## 10. CONNECT TO THE VNC SERVER
Create a new VNC server, in your selected client, where the `hostname` and the `port` are the output of `$NGROK_URL`
Example of an output of the command: `htto://4.tcp.ngrok.io:18462`, which means that the `hostname`is `4.tcp.ngrok.io` and the `port` is `18462`

## 11. RUN PYTHON APP
```
export DISPLAY=:1 
nice -n 10 python3 gui_run_conversion.py
```