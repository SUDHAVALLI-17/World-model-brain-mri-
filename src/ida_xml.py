"""Parse IDA per-image metadata XMLs (downloaded alongside images) into a table.

Each full XML holds the subject's research group (diagnosis at screening), sex, APOE
alleles, the visit's clinical scores (MMSE, CDR, FAQ, NPI-Q), age at the scan, the
acquisition date and the scanner protocol (manufacturer, model, field strength).
Small stub XMLs inside the image folders carry no subject block and are skipped.

  python src/ida_xml.py --xml_dir data/raw/ADNI --out data/ida_meta.csv
"""
import argparse
import glob
import os
import xml.etree.ElementTree as ET

import pandas as pd


def _txt(node, path):
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else None


def parse_xml(path):
    root = ET.parse(path).getroot()
    subj = root.find(".//subject")
    if subj is None or subj.find("subjectIdentifier") is None:
        return None
    rec = {"subject": _txt(subj, "subjectIdentifier"), "group_raw": _txt(subj, "researchGroup"),
           "sex": _txt(subj, "subjectSex")}
    for info in subj.findall("subjectInfo"):
        rec[info.get("item").replace(" ", "_").lower()] = info.text
    visit = subj.find("visit")
    if visit is not None:
        rec["visit"] = _txt(visit, "visitIdentifier")
        for s in visit.iter("assessmentScore"):
            rec[s.get("attribute").lower()] = float(s.text) if s.text else None
    study = subj.find("study")
    if study is not None:
        age = _txt(study, "subjectAge")
        rec["age"] = float(age) if age else None
        rec["acq_date"] = _txt(study, "series/dateAcquired")
        uid = _txt(study, "imagingProtocol/imageUID")
        rec["image_id"] = f"I{uid}" if uid else None
        rec["description"] = _txt(study, "imagingProtocol/description")
        for p in study.findall("imagingProtocol/protocolTerm/protocol"):
            term = p.get("term")
            if term in ("Manufacturer", "Mfg Model", "Field Strength", "Weighting", "Acquisition Type",
                        "Acquisition Plane", "Slice Thickness"):
                rec[term.lower().replace(" ", "_")] = p.text
    return rec


def parse_dir(xml_dir):
    rows = [r for f in glob.glob(os.path.join(xml_dir, "**", "*.xml"), recursive=True)
            if (r := parse_xml(f)) is not None]
    return pd.DataFrame(rows).drop_duplicates("image_id")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_dir", required=True)
    ap.add_argument("--out", default="data/ida_meta.csv")
    a = ap.parse_args()
    df = parse_dir(a.xml_dir)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} images, {df.subject.nunique()} subjects -> {a.out}")
