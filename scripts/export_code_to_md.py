import os
import json
import hashlib

OUTPUT_FILE = "PROJECT_CODE_DUMP.md"
SNAPSHOT_FILE = ".codebase_snapshot.json"

INCLUDE_EXTENSIONS = [".py", ".html", ".js", ".css", ".json"]

EXCLUDE_FOLDERS = [
    ".git",
    "__pycache__",
    "venv",
    "node_modules",
    "output"
]


def should_include_file(filename):
    return any(filename.endswith(ext) for ext in INCLUDE_EXTENSIONS)


def should_exclude_path(path):
    return any(excluded in path for excluded in EXCLUDE_FOLDERS)


# ---------- Analytics Helpers ----------

def file_hash(content):
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_snapshot():
    if not os.path.exists(SNAPSHOT_FILE):
        return {}
    try:
        with open(SNAPSHOT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}


def save_snapshot(snapshot):
    with open(SNAPSHOT_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)


# ---------- Main Export ----------

def export_codebase(root_dir="."):
    previous_snapshot = load_snapshot()
    current_snapshot = {}

    # Analytics counters
    new_files = []
    changed_files = []
    deleted_files = []
    total_added_lines = 0
    total_deleted_lines = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("# 📦 Project Full Code Dump\n\n")

        for root, dirs, files in os.walk(root_dir):
            if should_exclude_path(root):
                continue

            for file in files:
                if not should_include_file(file):
                    continue

                filepath = os.path.join(root, file)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    content = f"# Error reading file: {e}"

                lines = content.count("\n") + 1 if content else 0
                h = file_hash(content)

                current_snapshot[filepath] = {
                    "lines": lines,
                    "hash": h
                }

                # ---------- Analytics Comparison ----------
                if filepath not in previous_snapshot:
                    new_files.append(filepath)
                    total_added_lines += lines
                else:
                    old = previous_snapshot[filepath]

                    if old["hash"] != h:
                        changed_files.append(filepath)

                        diff = lines - old["lines"]
                        if diff > 0:
                            total_added_lines += diff
                        else:
                            total_deleted_lines += abs(diff)

                # ---------- Existing Dump Functionality ----------
                out.write(f"\n---\n")
                out.write(f"## 📄 {filepath}\n\n")
                out.write("```")

                ext = file.split(".")[-1]
                out.write(ext + "\n")
                out.write(content)
                out.write("\n```\n")

    # Detect deleted files
    for old_file in previous_snapshot:
        if old_file not in current_snapshot:
            deleted_files.append(old_file)
            total_deleted_lines += previous_snapshot[old_file]["lines"]

    save_snapshot(current_snapshot)

    # ---------- Terminal Analytics ----------
    print("\n📊 CODEBASE ANALYTICS")
    print("=" * 40)

    print(f"🆕 New Files: {len(new_files)}")
    for f in new_files[:5]:
        print("   +", f)
    if len(new_files) > 5:
        print("   ...")

    print(f"\n📝 Changed Files: {len(changed_files)}")
    for f in changed_files[:5]:
        print("   ~", f)
    if len(changed_files) > 5:
        print("   ...")

    print(f"\n🗑 Deleted Files: {len(deleted_files)}")
    for f in deleted_files[:5]:
        print("   -", f)
    if len(deleted_files) > 5:
        print("   ...")

    print("\n📈 Line Changes")
    print(f"   ➕ Lines Added: {total_added_lines}")
    print(f"   ➖ Lines Deleted: {total_deleted_lines}")

    print("\n✅ Code exported to", OUTPUT_FILE)


if __name__ == "__main__":
    export_codebase()
