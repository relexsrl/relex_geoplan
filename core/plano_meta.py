"""Per-plano metadata: filename parsing, sidecar persistence, cca assembly.

The plano metadata (nmp = registro number, cadastral nomenclatura components,
fecha) is captured once per raster and stored in a ``<tiff>.meta.json`` sidecar.
The filename pattern ``NNNNNN#DD-MM-SSS-CCCC-MMMM#...`` is only a best-effort
pre-fill; the authoritative values are printed on the plano itself (registro
stamp + NOMENCLATURA CATASTRAL table) and remain user-editable. Parsing stops
after the nomenclatura (the cca prefix); anything after it — including the plano
date — is intentionally ignored (the date is no longer used).
"""
import json
import re

# cca = dep(2) + mun(2) + sec(3) + chac(4) + mz(4) + etiqueta(4) = 19 chars
CCA_COMPONENTS = (("dep", 2), ("mun", 2), ("sec", 3), ("chac", 4), ("mz", 4))
CCA_PREFIX_LEN = sum(w for _key, w in CCA_COMPONENTS)  # 15
CCA_LEN = CCA_PREFIX_LEN + 4                            # prefix + 4-char etiqueta
META_KEYS = ("nmp", "dep", "mun", "sec", "chac", "mz", "fecha")

_FILENAME_RE = re.compile(
    r"^(?P<nmp>\d+)#"
    r"(?P<dep>\d{2})-(?P<mun>\d{2})-(?P<sec>\d{3})-(?P<chac>\d{4})-(?P<mz>\d{4})"
    r"(?=#|$)"  # bound the block number; stop here — don't parse the date
)
def parse_filename(name):
    """Pre-fill metadata from a plano filename; {} when it doesn't match.
    Only nmp + nomenclatura (through the cca prefix) are parsed; anything after
    the block number, including the plano date, is ignored."""
    m = _FILENAME_RE.match(name)
    if not m:
        return {}
    return {
        "nmp": m.group("nmp").lstrip("0") or "0",
        "dep": m.group("dep"),
        "mun": m.group("mun"),
        "sec": m.group("sec"),
        "chac": m.group("chac"),
        "mz": m.group("mz"),
    }


def sidecar_path(tiff_path):
    return str(tiff_path) + ".meta.json"


def load_sidecar(tiff_path):
    try:
        with open(sidecar_path(tiff_path), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return {k: str(data.get(k, "")).strip() for k in META_KEYS if data.get(k)}


def save_sidecar(tiff_path, meta):
    clean = {k: str(meta.get(k, "")).strip() for k in META_KEYS if str(meta.get(k, "")).strip()}
    if not clean:
        return False
    try:
        with open(sidecar_path(tiff_path), "w", encoding="utf-8") as fh:
            json.dump(clean, fh, indent=1)
        return True
    except OSError:
        return False


def normalize_etiqueta(etiqueta):
    """'18' -> '0018', '165a' -> '165A' (DB stores 4-char zero-padded, alnum ok)."""
    et = str(etiqueta or "").strip().upper()
    return et.zfill(4) if et else ""


def etiqueta_valid(etiqueta):
    """A real etiqueta is 4 chars: 3 digits + a final digit-or-letter, AND its
    parcel NUMBER is nonzero. The letter is a subdivision suffix of a numbered
    parcel, so '000A' and '0000' aren't parcels (verified on the DB: smallest
    number is 1). Accepts display forms (leading zeros dropped)."""
    et = normalize_etiqueta(etiqueta)
    if len(et) != 4 or not et[:3].isdigit() or not et[3].isalnum():
        return False
    number = et if et[3].isdigit() else et[:3]
    return int(number) >= 1


def build_codigo(meta):
    """15-char manzana codigo (= the cca prefix, dep+mun+sec+chac+mz) from the
    nomenclatura components; '' if incomplete."""
    parts = []
    for key, width in CCA_COMPONENTS:
        v = str(meta.get(key, "")).strip()
        if not v.isdigit() or len(v) > width:
            return ""
        parts.append(v.zfill(width))
    return "".join(parts)


def build_cca(meta, etiqueta):
    """19-char cca from the nomenclatura components + etiqueta; '' if incomplete."""
    et = normalize_etiqueta(etiqueta)
    if not et or len(et) > 4:
        return ""
    prefix = build_codigo(meta)
    return prefix + et if prefix else ""


