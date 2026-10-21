"""Masaüstü kısayolu ve oturum açılışında otomatik başlatma.

    ./run.sh --kisayol                  # Linux: masaüstü + uygulama menüsü
    ./run.sh --kisayol --otomatik-baslat
    run.bat --kisayol [--otomatik-baslat]   # Windows
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

from .config import ROOT

APP_NAME = "TI-RADS AI"
ICON = ROOT / "frontend" / "icon.svg"


def _linux_desktop_dir() -> Path:
    try:
        out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True,
                             text=True, timeout=3).stdout.strip()
        if out and Path(out).is_dir():
            return Path(out)
    except (OSError, subprocess.SubprocessError):
        pass
    for name in ("Masaüstü", "Desktop"):
        p = Path.home() / name
        if p.is_dir():
            return p
    return Path.home()


def _linux_entry() -> str:
    run = ROOT / "run.sh"
    return "\n".join([
        "[Desktop Entry]",
        "Type=Application",
        f"Name={APP_NAME}",
        "Comment=USG capture + UNet++ + ACR TI-RADS",
        f'Exec="{run}"',
        f"Path={ROOT}",
        f"Icon={ICON}",
        # Terminal penceresi açık kalır: programın çalıştığını gösterir,
        # kapatınca program durur.
        "Terminal=true",
        "Categories=Science;MedicalSoftware;",
        "",
    ])


def _write_desktop(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_linux_entry(), encoding="utf-8")
    path.chmod(0o755)
    # GNOME: masaüstündeki dosyayı "güvenilir" işaretle (yoksa çift tıkta sorar)
    try:
        subprocess.run(["gio", "set", str(path), "metadata::trusted", "true"],
                       capture_output=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        pass


def _win_shortcut(lnk: Path):
    run = ROOT / "run.bat"
    ps = (
        "$s=(New-Object -ComObject WScript.Shell).CreateShortcut($env:LNK);"
        "$s.TargetPath=$env:TARGET;$s.WorkingDirectory=$env:WD;"
        "$s.Description='USG capture + UNet++ + ACR TI-RADS';"
        "$s.WindowStyle=7;$s.Save()"
    )
    env = {**os.environ, "LNK": str(lnk), "TARGET": str(run), "WD": str(ROOT)}
    r = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                        "-Command", ps], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "Kısayol oluşturulamadı")


def _win_folder(name: str) -> Path:
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        f"[Environment]::GetFolderPath('{name}')"],
                       capture_output=True, text=True)
    return Path(r.stdout.strip())


def install(autostart: bool = False) -> list[Path]:
    made = []
    system = platform.system()
    if system == "Linux":
        desk = _linux_desktop_dir() / "tirads-ai.desktop"
        menu = Path.home() / ".local/share/applications/tirads-ai.desktop"
        for p in (desk, menu):
            _write_desktop(p)
            made.append(p)
        if autostart:
            a = Path.home() / ".config/autostart/tirads-ai.desktop"
            _write_desktop(a)
            made.append(a)
    elif system == "Windows":
        lnk = _win_folder("Desktop") / f"{APP_NAME}.lnk"
        _win_shortcut(lnk)
        made.append(lnk)
        if autostart:
            s = _win_folder("Startup") / f"{APP_NAME}.lnk"
            _win_shortcut(s)
            made.append(s)
    else:
        raise RuntimeError("Kısayol yalnızca Linux ve Windows için destekleniyor.")
    return made


def remove() -> list[Path]:
    gone = []
    system = platform.system()
    if system == "Linux":
        cands = [_linux_desktop_dir() / "tirads-ai.desktop",
                 Path.home() / ".local/share/applications/tirads-ai.desktop",
                 Path.home() / ".config/autostart/tirads-ai.desktop"]
    elif system == "Windows":
        cands = [_win_folder("Desktop") / f"{APP_NAME}.lnk",
                 _win_folder("Startup") / f"{APP_NAME}.lnk"]
    else:
        cands = []
    for p in cands:
        if p.exists():
            p.unlink()
            gone.append(p)
    return gone
