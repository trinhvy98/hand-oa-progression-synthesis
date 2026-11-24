"""Metadata parsing from hand.xlsx and image-level manifest construction."""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd

ALL_JOINTS = (
    "DIP2",
    "DIP3",
    "DIP4",
    "DIP5",
    "PIP2",
    "PIP3",
    "PIP4",
    "PIP5",
    "MCP2",
    "MCP3",
    "MCP4",
    "MCP5",
)

_NS_MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NS_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _read_xlsx_via_stdlib(excel_path: Path, sheet_name: str = "hand") -> pd.DataFrame:
    """Minimal zero-dependency XLSX reader for environments without openpyxl."""
    with zipfile.ZipFile(excel_path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            ss_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in ss_root.findall(f"{{{_NS_MAIN}}}si"):
                shared_strings.append(
                    "".join(t.text or "" for t in si.iter(f"{{{_NS_MAIN}}}t"))
                )

        wb_root = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"] for rel in rels_root
        }

        target_xml = None
        for sheet in wb_root.iter(f"{{{_NS_MAIN}}}sheet"):
            if sheet.attrib.get("name") == sheet_name:
                rid = sheet.attrib.get(f"{{{_NS_REL}}}id")
                target = rel_targets.get(rid, "").lstrip("/")
                target_xml = target if target.startswith("xl/") else f"xl/{target}"
                break
        if target_xml is None:
            target_xml = "xl/worksheets/sheet1.xml"

        sheet_root = ET.fromstring(archive.read(target_xml))
        rows = list(sheet_root.iter(f"{{{_NS_MAIN}}}row"))
        if not rows:
            return pd.DataFrame()

        def _parse_row(row_elem: ET.Element) -> dict[str, str]:
            parsed: dict[str, str] = {}
            for cell in row_elem.findall(f"{{{_NS_MAIN}}}c"):
                ref = cell.attrib.get("r", "")
                col_letters = "".join(ch for ch in ref if ch.isalpha())
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    is_elem = cell.find(f"{{{_NS_MAIN}}}is")
                    val = "".join(
                        t.text or "" for t in is_elem.iter(f"{{{_NS_MAIN}}}t")
                    ) if is_elem is not None else ""
                else:
                    v_elem = cell.find(f"{{{_NS_MAIN}}}v")
                    val = v_elem.text if v_elem is not None and v_elem.text is not None else ""
                    if cell_type == "s" and val != "":
                        val = shared_strings[int(val)]
                parsed[col_letters] = val
            return parsed

        header_map = _parse_row(rows[0])
        col_order = list(header_map.keys())
        col_names = [header_map[col] for col in col_order]

        records: list[list[str | None]] = []
        for row_elem in rows[1:]:
            row_dict = _parse_row(row_elem)
            records.append([row_dict.get(col) if row_dict.get(col, "") != "" else None for col in col_order])

    return pd.DataFrame(records, columns=col_names)


def load_excel_table(excel_path: str | Path, sheet_name: str = "hand") -> pd.DataFrame:
    """Load the hand OA metadata spreadsheet, using openpyxl if installed or XML fallback."""
    path = Path(excel_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata Excel file not found: {path}")
    try:
        import openpyxl  # noqa: F401

        return pd.read_excel(path, sheet_name=sheet_name)
    except ImportError:
        return _read_xlsx_via_stdlib(path, sheet_name=sheet_name)


def _normalize_patient_id(raw_id: object) -> str:
    """Convert raw Excel patient ID values (e.g. '1234567.0') to clean string IDs."""
    if raw_id is None or pd.isna(raw_id):
        return ""
    text = str(raw_id).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _parse_optional_int(raw_val: object) -> int | None:
    """Parse numeric score strings/floats safely into integers."""
    if raw_val is None or pd.isna(raw_val):
        return None
    text = str(raw_val).strip()
    if text == "":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _joint_group(joint: str) -> str:
    """Extract anatomical joint group prefix ('DIP', 'PIP', or 'MCP')."""
    upper = joint.upper()
    for prefix in ("DIP", "PIP", "MCP"):
        if upper.startswith(prefix):
            return prefix
    return upper


def build_manifest_from_excel(
    excel_path: str | Path,
    images_dir: str | Path,
    joints: Iterable[str] = ("DIP2", "DIP3", "DIP4", "DIP5"),
    visit: str = "v00",
    valid_kl_grades: Iterable[int] = (0, 1, 2, 3, 4),
    sheet_name: str = "hand",
) -> pd.DataFrame:
    """Build a deduplicated per-joint manifest linking Finger Joints PNGs to metadata labels."""
    img_root = Path(images_dir)
    if not img_root.is_dir():
        raise FileNotFoundError(f"Images directory not found: {img_root}")

    raw_df = load_excel_table(excel_path, sheet_name=sheet_name).copy()
    if "id" not in raw_df.columns:
        raise KeyError("Expected 'id' column in metadata spreadsheet")

    raw_df["patient_id"] = raw_df["id"].map(_normalize_patient_id)
    raw_df = raw_df[raw_df["patient_id"] != ""]
    # Deduplicate exact/repeated patient rows so a patient never enters multiple splits
    raw_df = raw_df.drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)

    available_files = {p.name for p in img_root.iterdir() if p.is_file()}
    valid_kl_set = {int(g) for g in valid_kl_grades}
    selected_joints = [j.upper() for j in joints]

    records: list[dict[str, object]] = []
    for row in raw_df.to_dict(orient="records"):
        patient_id = str(row["patient_id"])
        hand_side = _parse_optional_int(row.get("Hand"))
        for joint in selected_joints:
            kl_col = f"{visit}{joint}_KL"
            if kl_col not in raw_df.columns:
                raise KeyError(f"Missing expected KL column '{kl_col}' in {excel_path}")

            kl_grade = _parse_optional_int(row.get(kl_col))
            if kl_grade is None or kl_grade not in valid_kl_set:
                continue

            filename = f"{patient_id}_{joint.lower()}.png"
            if filename not in available_files:
                continue

            records.append(
                {
                    "sample_id": f"{patient_id}_{joint.lower()}",
                    "patient_id": patient_id,
                    "joint": joint,
                    "joint_group": _joint_group(joint),
                    "hand": hand_side,
                    "kl_grade": kl_grade,
                    "op_score": _parse_optional_int(row.get(f"{visit}{joint}_OP")),
                    "jsn_score": _parse_optional_int(row.get(f"{visit}{joint}_JSN")),
                    "er_score": _parse_optional_int(row.get(f"{visit}{joint}_ER")),
                    "filename": filename,
                }
            )

    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise ValueError("Manifest construction produced 0 valid image-label records")
    return manifest.sort_values(["patient_id", "joint"]).reset_index(drop=True)
