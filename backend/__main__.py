"""Başlatma:  python -m backend  [--host 0.0.0.0] [--port 8765] [--demo]"""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser

import uvicorn

from .config import Config


def already_running(url: str) -> bool:
    import json
    import urllib.request
    try:
        # Hastane proxy'si tanımlı olsa bile yerel adrese doğrudan bağlan
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url + "/api/status", timeout=0.8) as r:
            return "capture" in json.loads(r.read().decode("utf-8"))
    except Exception:
        return False


def main():
    cfg = Config().get()
    ap = argparse.ArgumentParser(description="TI-RADS AI Platform")
    ap.add_argument("--host", default=cfg["server"]["host"],
                    help="Varsayılan 127.0.0.1 (yalnızca bu bilgisayar). "
                         "Ağdan erişim için 0.0.0.0 – KVKK'yı göz önünde "
                         "bulundurun.")
    ap.add_argument("--port", type=int, default=cfg["server"]["port"])
    ap.add_argument("--demo", action="store_true",
                    help="Sentetik görüntü kaynağı + demo segmentasyon")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--kisayol", action="store_true",
                    help="Masaüstü kısayolu oluştur ve çık")
    ap.add_argument("--otomatik-baslat", action="store_true",
                    help="--kisayol ile: oturum açılınca otomatik başlat")
    ap.add_argument("--kisayol-kaldir", action="store_true")
    args = ap.parse_args()

    if args.kisayol or args.kisayol_kaldir:
        from . import shortcuts
        if args.kisayol_kaldir:
            gone = shortcuts.remove()
            print("Kaldırıldı:\n  " + "\n  ".join(map(str, gone)) if gone
                  else "Kaldırılacak kısayol yok.")
        else:
            made = shortcuts.install(autostart=args.otomatik_baslat)
            print("Oluşturuldu:\n  " + "\n  ".join(map(str, made)))
        return

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    # Program zaten çalışıyorsa ikinci kez başlatma: yalnızca tarayıcıyı aç
    if already_running(url):
        print(f"TI-RADS AI zaten çalışıyor: {url}")
        if not args.no_browser:
            webbrowser.open(url)
        return

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from .server import create_app
    # --demo ayar dosyasını değiştirmez; yalnızca bu oturum için geçerlidir.
    app = create_app(demo=args.demo)

    print(f"\n  TI-RADS AI Platform çalışıyor: {url}\n  Durdurmak için Ctrl+C\n")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
