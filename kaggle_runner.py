#!/usr/bin/env python3
import sys
import os
import json
import time
import uuid
import ssl
import urllib.request
import urllib.parse
import re
try:
    import websocket
except ImportError:
    websocket = None

PROXY_URL = os.environ.get(
    "KAGGLE_PROXY_URL",
    "https://kkb-production.jupyter-proxy.kaggle.net/k/353880983/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..ZSyM02bfO4bnNPL9-fR8lA.2WoFReKIrqoIDtYiSJL5tOUXqlDCdEYs5qbBAEnvqSFYmAZiQkZrz6PDbsbV84DagaCk1Een-zrWDjl974PtECF-JFUCyNsdWuY8HkF4ufRRN1X3feTLeF_WlQJt-4jHYyGPRUZ7ctqLJzik1VGT9nsiiAZQjRj6CZ2REJM79w5AxtEnXn5vd1PoGvKQM3rKkgDYcNX4VcxrgHrAs9B65zjx5AVyn_L5ReLlqkqzmNp7IE7py4kOYg2J3BseWCnn.YNRQd4zkRHN8De6YGnri3Q/proxy",
)

def check_and_update_proxy(resp_headers):
    global PROXY_URL
    cookie = resp_headers.get("set-cookie", "")
    report = resp_headers.get("report-uri", "")
    m = re.search(r"(/k/\d+/[A-Za-z0-9_\-\.]+/proxy)", cookie + " " + report)
    if m:
        new_url = "https://kkb-production.jupyter-proxy.kaggle.net" + m.group(1)
        if new_url != PROXY_URL:
            print(f"[PROXY AUTO-HEAL] Updating PROXY_URL -> {new_url}")
            PROXY_URL = new_url

import base64
import socket
import struct

class SimpleWebSocket:
    def __init__(self, host, path, timeout=15):
        context = ssl.create_default_context()
        sock = socket.create_connection((host, 443), timeout=timeout)
        self.sock = context.wrap_socket(sock, server_hostname=host)
        self.sock.settimeout(timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        handshake = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"Origin: https://kkb-production.jupyter-proxy.kaggle.net\r\n\r\n"
        )
        self.sock.sendall(handshake.encode())
        resp = self.sock.recv(4096).decode("latin1", errors="replace")
        if "101" not in resp.splitlines()[0]:
            raise RuntimeError(f"WebSocket handshake failed: {resp[:200]}")

    def send(self, text):
        data = text.encode("utf-8")
        length = len(data)
        mask_key = os.urandom(4)
        masked_data = bytes(b ^ mask_key[i % 4] for i, b in enumerate(data))
        if length < 126:
            header = struct.pack("!BB", 0x81, 0x80 | length)
        elif length < 65536:
            header = struct.pack("!BBH", 0x81, 0x80 | 126, length)
        else:
            header = struct.pack("!BBQ", 0x81, 0x80 | 127, length)
        self.sock.sendall(header + mask_key + masked_data)

    def _recv_exact(self, n):
        buf = b""
        while len(buf) < n:
            try:
                chunk = self.sock.recv(n - len(buf))
                if not chunk:
                    return None
                buf += chunk
            except (socket.timeout, TimeoutError):
                if not buf:
                    return b""
                continue
        return buf

    def recv(self):
        try:
            head = self._recv_exact(2)
            if head is None:
                return None
            if not head:
                return ""
            b1, b2 = struct.unpack("!BB", head)
            opcode = b1 & 0x0F
            has_mask = bool(b2 & 0x80)
            length = b2 & 0x7F
            if length == 126:
                ext = self._recv_exact(2)
                if not ext or len(ext) < 2:
                    return None
                length = struct.unpack("!H", ext)[0]
            elif length == 127:
                ext = self._recv_exact(8)
                if not ext or len(ext) < 8:
                    return None
                length = struct.unpack("!Q", ext)[0]
            mask_key = self._recv_exact(4) if has_mask else None
            data = self._recv_exact(length) if length > 0 else b""
            if data is None:
                return None
            if has_mask and mask_key:
                data = bytes(b ^ mask_key[i % 4] for i, b in enumerate(data))
            if opcode == 0x1:
                return data.decode("utf-8", errors="replace")
            elif opcode == 0x9:  # ping
                self.sock.sendall(struct.pack("!BB", 0x8A, 0x00))
                return self.recv()
            elif opcode == 0x8:  # close
                return None
            return ""
        except (socket.timeout, TimeoutError):
            return ""

    def settimeout(self, timeout):
        self.sock.settimeout(timeout)

    def ping(self):
        self.sock.sendall(struct.pack("!BB", 0x89, 0x80) + os.urandom(4))

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

def get_kernel_id():
    global PROXY_URL
    explicit_id = os.environ.get("KAGGLE_KERNEL_ID")
    if explicit_id:
        return explicit_id
    for _ in range(2):
        try:
            # Check sessions first to find the notebook kernel
            req_s = urllib.request.Request(f"{PROXY_URL}/api/sessions")
            with urllib.request.urlopen(req_s, timeout=15) as resp:
                check_and_update_proxy(resp.headers)
                sessions = json.loads(resp.read().decode())
                for s in sessions:
                    if "krist" in s.get("path", "").lower() and s.get("kernel", {}).get("id"):
                        return s["kernel"]["id"]
                if sessions and sessions[0].get("kernel", {}).get("id"):
                    return sessions[0]["kernel"]["id"]
            
            req = urllib.request.Request(f"{PROXY_URL}/api/kernels")
            with urllib.request.urlopen(req, timeout=15) as resp:
                check_and_update_proxy(resp.headers)
                kernels = json.loads(resp.read().decode())
                if not kernels:
                    raise RuntimeError("No running kernels found on Kaggle session.")
                idles = [k for k in kernels if k.get("execution_state") == "idle"]
                if idles:
                    return idles[0]["id"]
                connected = [k for k in kernels if k.get("connections", 0) > 0]
                if connected:
                    return connected[0]["id"]
                return kernels[0]["id"]
        except urllib.error.HTTPError as e:
            check_and_update_proxy(e.headers)
    raise RuntimeError("Failed to obtain kernel_id after proxy refresh")

def execute_code(code: str, timeout: float = 1800.0, print_stream: bool = True, max_retries: int = 3):
    kernel_id = get_kernel_id()
    url = PROXY_URL.replace("https://", "").replace("http://", "")
    host = url.split("/")[0]
    path = "/" + "/".join(url.split("/")[1:]) + f"/api/kernels/{kernel_id}/channels"

    for attempt in range(1, max_retries + 1):
        try:
            ws = SimpleWebSocket(host, path, timeout=15)
            break
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(1.5 * attempt)
    
    msg_id = str(uuid.uuid4())
    session_id = str(uuid.uuid4())
    req = {
        "header": {
            "msg_id": msg_id,
            "username": "antigravity",
            "session": session_id,
            "msg_type": "execute_request",
            "version": "5.3",
        },
        "parent_header": {},
        "metadata": {},
        "content": {
            "code": code,
            "silent": False,
            "store_history": True,
            "user_expressions": {},
            "allow_stdin": False,
            "stop_on_error": True,
        },
        "channel": "shell",
    }
    ws.send(json.dumps(req))
    
    output_chunks = []
    start_time = time.time()
    ws.settimeout(5.0)
    
    while time.time() - start_time < timeout:
        try:
            raw = ws.recv()
            if not raw:
                continue
            msg = json.loads(raw)
            msg_type = msg.get("header", {}).get("msg_type")
            parent_id = msg.get("parent_header", {}).get("msg_id")
            
            if parent_id != msg_id:
                continue
                
            if msg_type == "stream":
                text = msg["content"]["text"]
                output_chunks.append(text)
                if print_stream:
                    sys.stdout.write(text)
                    sys.stdout.flush()
            elif msg_type == "error":
                err_text = "\n".join(msg["content"]["traceback"]) + "\n"
                output_chunks.append(err_text)
                if print_stream:
                    sys.stderr.write(err_text)
                    sys.stderr.flush()
            elif msg_type == "execute_result":
                res_text = str(msg["content"]["data"].get("text/plain", "")) + "\n"
                output_chunks.append(res_text)
                if print_stream:
                    sys.stdout.write(res_text)
                    sys.stdout.flush()
            elif msg_type == "status" and msg["content"].get("execution_state") == "idle" and parent_id == msg_id:
                break
        except (socket.timeout, TimeoutError):
            continue

        except Exception as e:
            err_msg = f"\n[WebSocket Error: {e}]\n"
            output_chunks.append(err_msg)
            if print_stream:
                sys.stderr.write(err_msg)
                sys.stderr.flush()
            break
            
    try:
        ws.close()
    except Exception:
        pass
    return "".join(output_chunks)

def push_file(local_path: str, remote_rel_path: str = None):
    if remote_rel_path is None:
        remote_rel_path = os.path.basename(local_path)
    if local_path.endswith(".ipynb"):
        with open(local_path, "r", encoding="utf-8") as f:
            nb_content = json.load(f)
        payload = {"type": "notebook", "format": "json", "content": nb_content}
    else:
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()
            payload = {"content": content, "format": "text", "type": "file"}
        except UnicodeDecodeError:
            with open(local_path, "rb") as f:
                content = base64.b64encode(f.read()).decode("ascii")
            payload = {"content": content, "format": "base64", "type": "file"}
    encoded = urllib.parse.quote(remote_rel_path)
    url = f"{PROXY_URL}/api/contents/{encoded}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="PUT")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status in (200, 201):
            print(f"[PUSH SUCCESS] {local_path} -> {remote_rel_path}")
        else:
            print(f"[PUSH STATUS {resp.status}] {remote_rel_path}")

def pull_file(remote_rel_path: str, local_path: str = None):
    if local_path is None:
        local_path = os.path.basename(remote_rel_path)
    encoded = urllib.parse.quote(remote_rel_path)
    url = f"{PROXY_URL}/api/contents/{encoded}?content=1"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status == 200:
            res = json.loads(resp.read().decode())
            content = res.get("content", "")
            fmt = res.get("format", "")
            if fmt == "base64" and isinstance(content, str):
                with open(local_path, "wb") as f:
                    f.write(base64.b64decode(content))
            else:
                with open(local_path, "w", encoding="utf-8") as f:
                    if isinstance(content, dict):
                        json.dump(content, f, indent=1)
                    else:
                        f.write(content)
            print(f"[PULL SUCCESS] {remote_rel_path} -> {local_path}")
        else:
            print(f"[PULL STATUS {resp.status}] {remote_rel_path}")

def push_via_exec(local_path: str, remote_path: str = None):
    if remote_path is None:
        remote_path = os.path.basename(local_path)
    with open(local_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    chunk_size = 2_000_000
    if len(b64) <= chunk_size:
        code = f"""import base64
with open('{remote_path}', 'wb') as f:
    f.write(base64.b64decode('{b64}'))
print(f'[PUSH SUCCESS] {remote_path} written via websocket ({len(b64)} b64 chars)')
"""
        execute_code(code)
    else:
        execute_code(f"with open('{remote_path}', 'wb') as f: pass")
        for i in range(0, len(b64), chunk_size):
            chunk = b64[i : i + chunk_size]
            code = f"""import base64
with open('{remote_path}', 'ab') as f:
    f.write(base64.b64decode('{chunk}'))
"""
            execute_code(code)
        print(f"[PUSH SUCCESS] {remote_path} written in chunks ({len(b64)} b64 chars)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "exec":
            code = " ".join(sys.argv[2:])
            execute_code(code)
        elif cmd == "push":
            push_via_exec(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        elif cmd == "pull":
            pull_file(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        elif cmd == "status":
            execute_code("import sys, os\nprint('Python:', sys.version.split()[0])\nprint('Working Dir:', os.listdir('.'))")
    else:
        print("Usage: kaggle_runner.py [exec <code> | push <local> [remote] | pull <remote> [local] | status]")
