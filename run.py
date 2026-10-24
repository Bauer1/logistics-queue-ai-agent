import socket
import sys
import webbrowser
import uvicorn


def is_port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    if is_port_in_use(host, port):
        print("=================================================================")
        print(f"[Logistics AI-Agent] Сервер уже запущен и работает на {url}")
        print("Открываю дашборд в браузере...")
        print("=================================================================")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        sys.exit(0)

    print("=================================================================")
    print("[Logistics AI-Agent] Starting Queue & Dock Optimization Server...")
    print(f"[Web Dashboard] Open in browser: {url}")
    print("=================================================================")
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)
