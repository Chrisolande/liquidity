#!/usr/bin/env python3
"""
Continuous IOPub WebSocket monitor for the running Kaggle kernel.
Streams all output to stdout/log, sends periodic pings, and auto-reconnects
until the kernel enters idle state.
"""
import json
import os
import sys
import time
import urllib.request

from kaggle_runner import PROXY_URL, SimpleWebSocket, get_kernel_id


def check_kernel_idle(kid: str) -> bool:
    try:
        req = urllib.request.Request(f"{PROXY_URL}/api/kernels/{kid}")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("execution_state") == "idle"
    except Exception:
        return False


def monitor_loop(log_file: str = "/tmp/tabpfn_monitor.log"):
    kid = get_kernel_id()
    url = PROXY_URL.replace("https://", "").replace("http://", "")
    host = url.split("/")[0]
    path = "/" + "/".join(url.split("/")[1:]) + f"/api/kernels/{kid}/channels"

    print(f"[{time.strftime('%H:%M:%S')}] Monitoring Kaggle Kernel {kid} via WebSocket...", flush=True)

    with open(log_file, "a", encoding="utf-8") as lf:
        while True:
            if check_kernel_idle(kid):
                msg = f"\n[{time.strftime('%H:%M:%S')}] Kernel execution state is IDLE. Benchmark completed.\n"
                print(msg, flush=True)
                lf.write(msg)
                lf.flush()
                break

            try:
                ws = SimpleWebSocket(host, path, timeout=10)
                last_ping = time.time()
                while True:
                    raw = ws.recv()
                    if raw:
                        try:
                            msg = json.loads(raw)
                            mtype = msg.get("header", {}).get("msg_type")
                            if mtype == "stream":
                                text = msg.get("content", {}).get("text", "")
                                sys.stdout.write(text)
                                sys.stdout.flush()
                                lf.write(text)
                                lf.flush()
                            elif mtype == "error":
                                err = "\n".join(msg.get("content", {}).get("traceback", [])) + "\n"
                                sys.stderr.write(err)
                                sys.stderr.flush()
                                lf.write(err)
                                lf.flush()
                            elif mtype == "status" and msg.get("content", {}).get("execution_state") == "idle":
                                msg_txt = f"\n[{time.strftime('%H:%M:%S')}] Received idle status from kernel!\n"
                                print(msg_txt, flush=True)
                                lf.write(msg_txt)
                                lf.flush()
                                return
                        except Exception:
                            pass

                    # Periodic keepalive ping every 20 seconds
                    if time.time() - last_ping > 20:
                        try:
                            ws.ping()
                        except Exception:
                            break
                        last_ping = time.time()

                    # Periodic idle check every 60 seconds
                    if time.time() - last_ping > 60:
                        if check_kernel_idle(kid):
                            return

            except Exception as e:
                time.sleep(3)
                if check_kernel_idle(kid):
                    return


if __name__ == "__main__":
    monitor_loop()
