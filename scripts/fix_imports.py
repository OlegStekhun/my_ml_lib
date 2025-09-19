"""
scripts/fix_imports.py

Надёжный скрипт для исправления внутрипакетных импортов:
заменяет "from module import X" на "from .module import X" для модулей внутри пакета.

Запускать из корня проекта:
    python scripts/fix_imports.py
"""

import re
import sys
from pathlib import Path

def find_package_dir():
    """
    Ищем папку пакета в следующем порядке:
    1) <repo_root>/my_ml_lib
    2) Любая папка уровня repo_root с файлом __init__.py (топ-уровень)
    3) Любая папка (глубже) с __init__.py (исключая виртуальные среды)
    """
    repo_root = Path(__file__).resolve().parent.parent
    # 1) предпочтительная точная папка
    pref = repo_root / "my_ml_lib"
    if pref.exists() and pref.is_dir():
        return pref, repo_root

    # 2) ищем папки с __init__.py на одном уровне
    for init in repo_root.glob("*/__init__.py"):
        return init.parent, repo_root

    # 3) ищем глубже, но пропускаем обычные виртуальные окружения и скрытые папки
    for init in repo_root.glob("**/__init__.py"):
        parts = set(init.parts)
        if ".venv" in parts or "venv" in parts or "env" in parts or "__pycache__" in parts:
            continue
        return init.parent, repo_root

    return None, repo_root

def fix_imports(pkg_dir: Path, repo_root: Path):
    internal_modules = [p.stem for p in pkg_dir.glob("*.py") if p.stem != "__init__"]

    py_files = list(pkg_dir.glob("*.py"))
    if not py_files:
        print(f"⚠️ В папке пакета нет .py файлов: {pkg_dir}")
        return

    for file in py_files:
        text = file.read_text(encoding="utf-8")
        original = text
        for mod in internal_modules:
            # Заменяем только полноценные "from <mod> ..." (начало строки с возможными отступами)
            text = re.sub(
                rf'(^\s*)from\s+{re.escape(mod)}\b',
                rf'\1from .{mod}',
                text,
                flags=re.M,
            )
        if text != original:
            file.write_text(text, encoding="utf-8")
            try:
                rel = file.relative_to(repo_root)
            except Exception:
                rel = file
            print(f"✔ Patched {rel}")

if __name__ == "__main__":
    pkg_dir, repo_root = find_package_dir()
    if not pkg_dir:
        repo_root = Path(__file__).resolve().parent.parent
        print("❌ Папка пакета не найдена.")
        print(f"Искал: папку 'my_ml_lib' или любую папку с __init__.py внутри {repo_root}")
        print("Содержимое корня проекта:")
        for p in sorted(repo_root.iterdir()):
            print("  -", p.name)
        sys.exit(1)

    print(f"ℹ️ Найдена папка пакета: {pkg_dir}")
    fix_imports(pkg_dir, repo_root)
    print("✅ Готово.")
