import math
import os
import sys
import tempfile
import traceback
from datetime import datetime

from qgis.PyQt.QtCore import QCoreApplication, QEvent, QRegularExpression, QSettings, Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import (
    QColor,
    QIcon,
    QImage,
    QPixmap,
    QRegularExpressionValidator,
)
from qgis.PyQt.QtWidgets import (
    QAction,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsFillSymbol,
    QgsGeometry,
    QgsLineSymbol,
    QgsMapLayer,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsSingleSymbolRenderer,
    QgsVectorFileWriter,
    QgsVectorDataProvider,
    QgsVectorLayer,
    QgsWkbTypes,
)
from qgis.gui import QgsMapTool, QgsRubberBand
from osgeo import gdal, osr

from .core.models import Geotransform, Point
from .core.plano_meta import CCA_COMPONENTS, build_cca, build_codigo, etiqueta_valid, load_sidecar, normalize_etiqueta, parse_filename, save_sidecar
from .i18n import install_translator, tr
# Heavy/optional dependencies (numpy, scipy, shapely, OpenCV).
# Imported once here instead of per-call; if any is missing the plugin still loads
# and ``run`` reports it via ``_MISSING_DEPENDENCY`` rather than failing to import.
try:
    from .raster_io import read_tiff
    from .core.cv2_pipeline import extract_with_cv2
    from .core.etiqueta_ocr import load_library, match_etiquetas, read_etiquetas
    from .core.char_reader import (
        AUTO_GATE, SUGGEST_GATE,
        load_reader as load_char_reader,
        read_polygons as read_char_polygons,
    )
    from .core.etiqueta_harvest import harvest_confirmed, harvest_disk, marker_disks
    from .core.etiqueta_ocr import crop_at_rect, disk_at_point
    from .core.etiqueta_order import assign_etiquetas
    from .core.roi import expand_roi, roi_from_world_polygon

    _MISSING_DEPENDENCY = None
except ImportError as exc:
    read_tiff = None
    extract_with_cv2 = None
    assign_etiquetas = None
    load_library = match_etiquetas = read_etiquetas = None
    load_char_reader = read_char_polygons = None
    AUTO_GATE = SUGGEST_GATE = 1.0
    harvest_confirmed = harvest_disk = marker_disks = crop_at_rect = disk_at_point = None
    expand_roi = roi_from_world_polygon = None
    _MISSING_DEPENDENCY = getattr(exc, "name", None) or "a required package"


def _numpy_runtime():
    try:
        import numpy as np
        return np.__version__, os.path.abspath(np.__file__)
    except Exception:
        return None, None


def _opencv_requirement():
    numpy_version, _origin = _numpy_runtime()
    numpy_major = int(numpy_version.split(".", 1)[0]) if numpy_version else None
    return (
        "opencv-contrib-python-headless>=4.13,<5"
        if numpy_major is not None and numpy_major >= 2
        else "opencv-contrib-python-headless>=4.8,<4.12" if numpy_major == 1 else None
    )


DEPENDENCY_PACKAGES = [
    "scipy>=1.7",
    "shapely>=1.8",
]
if _opencv_requirement() is not None:
    DEPENDENCY_PACKAGES.append(_opencv_requirement())


def _dependency_install_message(missing_module: str) -> str:
    packages = " ".join(f"'{package}'" for package in DEPENDENCY_PACKAGES)
    bin_dir = os.path.dirname(sys.executable)
    launchers = [
        os.path.join(bin_dir, name)
        for name in ("python-qgis-ltr.bat", "python-qgis.bat")
    ]
    launcher = next((path for path in launchers if os.path.isfile(path)), launchers[0])
    numpy_version, numpy_origin = _numpy_runtime()
    environment = (
        f"QGIS: {Qgis.QGIS_VERSION}\n"
        f"Python: {sys.version.split()[0]}\n"
        f"NumPy: {numpy_version or 'unavailable'}"
        + (f"\nNumPy path: {numpy_origin}" if numpy_origin else "")
    )
    numpy_note = (
        "QGIS supplies NumPy. If NumPy is the missing module, remove any pip-installed "
        "user copy instead of installing another one, or repair QGIS. The compatible "
        "OpenCV version cannot be selected until QGIS NumPy imports successfully.\n\n"
        if missing_module == "numpy" or numpy_version is None
        else ""
    )
    command = (
        f'& "{launcher}" -m pip install --user --no-deps {packages}'
        if numpy_version is not None
        else "Fix the QGIS NumPy import first, then reopen this message."
    )
    return (
        f"Missing Python module: {missing_module}\n\n"
        f"{environment}\n\n"
        f"{numpy_note}"
        "Close QGIS, run this command in PowerShell, then restart QGIS:\n\n"
        f"{command}\n\n"
        "Do not use sys.executable from the QGIS Python Console; it points to "
        "qgis-bin, not the Python launcher."
    )


def _ring_area(ring):
    if len(ring) < 3:
        return 0.0
    area = 0.0
    for a, b in zip(ring, ring[1:] + ring[:1]):
        area += a.x * b.y - b.x * a.y
    return abs(area) / 2.0


def _largest_polygons(polygons, expected_count):
    if expected_count <= 0 or len(polygons) <= expected_count:
        return polygons
    return sorted(polygons, key=_ring_area, reverse=True)[:expected_count]


def _largest_ring_area(result):
    return max((_ring_area(ring) for ring in result.get("polygons", [])), default=0.0)


def _expected_count(settings):
    """Resolve the expected parcel count from settings (0 = unknown / feature off)."""
    if not settings.get("use_expected_parcel_count"):
        return 0
    return int(settings.get("expected_parcel_count", 0) or 0)


def _parcel_fill_symbol():
    """Fresh red parcel fill symbol (a renderer takes ownership of its symbol)."""
    return QgsFillSymbol.createSimple(
        {"outline_width": "0.5", "outline_color": "255,0,0,255", "color": "255,100,100,128"}
    )


def _expected_count_score(polygons, expected_count):
    if expected_count <= 0:
        return (0, 0.0)
    selected = _largest_polygons(polygons, expected_count)
    enough = 1 if len(selected) >= expected_count else 0
    return (enough, sum(_ring_area(ring) for ring in selected))


def _with_largest_polygons(result, expected_count):
    trimmed = dict(result)
    trimmed["polygons"] = _largest_polygons(result.get("polygons", []), expected_count)
    return trimmed


def _is_writable_polygon_layer(layer):
    """Base test for a publish destination: a valid, file-backed, writable
    polygon layer."""
    if (
        layer is None
        or layer.type() != QgsMapLayer.VectorLayer
        or not layer.isValid()
        or layer.dataProvider() is None
        or layer.dataProvider().name() == "memory"
        or not layer.crs().isValid()
        or QgsWkbTypes.geometryType(layer.wkbType()) != QgsWkbTypes.PolygonGeometry
    ):
        return False
    return bool(layer.dataProvider().capabilities() & QgsVectorDataProvider.AddFeatures)


def _layer_kind(layer):
    return str(layer.customProperty("parcel_geometry/kind", "") or "").strip().lower()


def _layer_role(layer):
    return str(layer.customProperty("parcel_geometry/role", "") or "").strip().lower()


def _is_publish_destination_layer(layer):
    """Whether a loaded project layer can receive extracted geometry."""
    if not _is_writable_polygon_layer(layer):
        return False
    fields = layer.fields()
    return fields.indexOf("cca") >= 0 and fields.indexOf("etiqueta") >= 0


def _is_temporary_publish_layer(layer):
    return _is_temporary_parcel_layer(layer) or _is_temporary_manzana_layer(layer)


def _is_temporary_memory_polygon_layer(layer):
    return (
        layer is not None
        and layer.type() == QgsMapLayer.VectorLayer
        and layer.dataProvider() is not None
        and layer.dataProvider().name() == "memory"
        and QgsWkbTypes.geometryType(layer.wkbType()) == QgsWkbTypes.PolygonGeometry
    )


def _is_temporary_parcel_layer(layer):
    if not _is_temporary_memory_polygon_layer(layer):
        return False
    if (_layer_role(layer) != "source"
            or _layer_kind(layer) != "parcela"):
        return False
    fields = layer.fields()
    return all(
        fields.indexOf(name) >= 0
        for name in (
            "id", "cca", "etiqueta", "sec", "mz", "reviewed",
            "etiqueta_suggestion", "nmp", "fecha",
        )
    )


def _is_temporary_manzana_layer(layer):
    if not _is_temporary_memory_polygon_layer(layer):
        return False
    if (_layer_role(layer) != "source"
            or _layer_kind(layer) != "manzana"):
        return False
    fields = layer.fields()
    return all(
        fields.indexOf(name) >= 0
        for name in ("id", "codigo", "dep", "mun", "sec", "chac", "mz", "nmp")
    ) and fields.indexOf("etiqueta") < 0


def _default_layer_name(meta):
    """Per-plano layer name from the plano's identity: nmp + the nomenclatura
    through manzana, e.g. '064286 · 13-55-010-0000-0234'. Falls back to a
    generic name when the metadata isn't known yet."""
    meta = meta or {}
    nmp = str(meta.get("nmp", "")).strip()
    parts = []
    for key, width in CCA_COMPONENTS:
        val = str(meta.get(key, "")).strip()
        if not val:
            parts = []
            break
        parts.append(val.zfill(width))
    nomen = "-".join(parts)
    return " · ".join(x for x in (nmp, nomen) if x) or "Parcel line candidates"


def _disk_to_pixmap(disk, height=46):
    """Grayscale marker-disk array -> QPixmap thumbnail (owned buffer)."""
    try:
        import numpy as np
        a = np.ascontiguousarray(disk, dtype=np.uint8)
        h, w = a.shape
        img = QImage(a.tobytes(), w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(img).scaledToHeight(height, Qt.SmoothTransformation)
    except Exception:
        return None


_GLYPH_SAMPLES = None
_CHAR_READER = False  # False = not yet loaded; None = unavailable; tuple = (net, classes)


def _bundled_glyph_samples():
    global _GLYPH_SAMPLES
    if _GLYPH_SAMPLES is None:
        _GLYPH_SAMPLES = load_library() if load_library is not None else []
    return _GLYPH_SAMPLES


def _bundled_char_reader():
    """Cached (net, classes) for the CNN suggestion reader, or None."""
    global _CHAR_READER
    if _CHAR_READER is False:
        _CHAR_READER = load_char_reader() if load_char_reader is not None else None
    return _CHAR_READER


def _merge_assignments(spatial, refined, confident, etiquetas):
    """OCR-confident labels win; remaining polygons keep their spatial label
    when still free, else take the next unused typed label."""
    n = len(spatial)
    merged = [refined[i] if confident[i] else "" for i in range(n)]
    used = {m.upper() for m in merged if m}
    pool = [l for l in etiquetas
            if str(l).strip() and str(l).strip().upper() not in used]
    for i in range(n):
        if merged[i]:
            continue
        cand = str(spatial[i]).strip() if spatial[i] else ""
        if cand and cand.upper() not in used and cand in pool:
            pool.remove(cand)
        elif pool:
            cand = pool.pop(0)
        else:
            cand = ""
        merged[i] = cand
        if cand:
            used.add(cand.upper())
    return merged


def _geom_exterior_points(geom):
    """QgsGeometry -> exterior ring as [Point], or [] (multi: largest part)."""
    if geom is None or geom.isEmpty():
        return []
    polys = geom.asMultiPolygon() if geom.isMultipart() else [geom.asPolygon()]
    polys = [p for p in polys if p and p[0]]
    if not polys:
        return []
    ring = max(polys, key=lambda p: len(p[0]))[0]
    if len(ring) > 1 and ring[0] == ring[-1]:
        ring = ring[:-1]
    return [Point(x=p.x(), y=p.y()) for p in ring]


def _ocr_crop(rings, gt, array):
    """Pixel crop + polygon coords for marker OCR. rings: world point lists."""
    det = gt.pixel_width * gt.pixel_height - gt.rot_x * gt.rot_y

    def to_px(x, y):
        dx, dy = x - gt.origin_x, y - gt.origin_y
        return ((dx * gt.pixel_height - dy * gt.rot_x) / det,
                (dy * gt.pixel_width - dx * gt.rot_y) / det)

    polys_px = [[to_px(p.x, p.y) for p in ring] for ring in rings]
    pad = 260
    cols = [c for poly in polys_px for c, _r in poly]
    rows = [r for poly in polys_px for _c, r in poly]
    x0 = max(0, int(min(cols)) - pad)
    y0 = max(0, int(min(rows)) - pad)
    x1 = min(array.shape[1], int(max(cols)) + pad)
    y1 = min(array.shape[0], int(max(rows)) + pad)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    crop = array[y0:y1, x0:x1]
    rel = [[(c - x0, r - y0) for c, r in poly] for poly in polys_px]
    return crop, rel


def _ocr_match_polygons(rings, etiquetas, gt, array, samples):
    """Match typed labels to polygons via the circled numbers."""
    prepared = _ocr_crop(rings, gt, array)
    if prepared is None:
        return [""] * len(rings), [False] * len(rings)
    crop, rel = prepared
    return match_etiquetas(crop, rel, etiquetas, samples)


def _ocr_read_polygons(rings, gt, array, samples):
    """Free-read each polygon's circled number straight off the raster. Returns
    (labels, confident, suggestions). Two readers, both held to the zero-wrong bar
    (a wrong etiqueta poisons the cca — blank beats wrong):
      - the template matcher (confident auto-reads), then
      - the CNN handwriting reader: at/above AUTO_GATE it becomes a confident
        auto-read too (measured zero-wrong on held-out sheets); in the weaker
        SUGGEST band it only fills a placeholder hint for the fill-in dialog."""
    prepared = _ocr_crop(rings, gt, array)
    if prepared is None:
        n = len(rings)
        return [""] * n, [False] * n, [""] * n
    crop, rel = prepared
    dbg = []
    labels, confident = read_etiquetas(crop, rel, samples, debug_scores=dbg)
    suggestions = [""] * len(rings)
    for i, best_text, _score in dbg:
        if 0 <= i < len(suggestions):
            suggestions[i] = best_text or ""
    reader = _bundled_char_reader()
    if reader is not None:
        try:
            net, classes = reader
            taken = {labels[i].upper() for i in range(len(labels))
                     if confident[i] and labels[i]}
            cnn = read_char_polygons(net, classes, crop, rel)
            for i, (text, conf) in enumerate(cnn):
                if not text or (confident[i] and labels[i]):
                    continue
                if conf >= AUTO_GATE and text.upper() not in taken:
                    labels[i] = text          # zero-wrong band: a real read
                    confident[i] = True
                    taken.add(text.upper())
                elif conf >= SUGGEST_GATE:
                    suggestions[i] = text      # hint the user confirms
        except Exception:
            pass
    return labels, confident, suggestions


def _assign_etiquetas_result(result, settings, geotransform, array):
    """Attach the per-polygon etiqueta assignment (spatial order + marker OCR)
    to the result. Runs in the WORKER thread — OCR can take a while and must
    never freeze the UI. ``assigned_etiquetas`` aligns with result['polygons']."""
    polys = result.get("polygons")
    if polys is None:
        return result
    if settings.get("manzana_mode"):
        return result  # a manzana outline has no etiquetas to assign/read
    etiquetas = settings.get("etiquetas") or []
    n = len(polys)
    assigned = [str(etiquetas[i]).strip() if i < len(etiquetas) else ""
                for i in range(n)]
    suggestions = [""] * n
    # provenance per polygon, for the harvest flywheel: "typed" (single parcel,
    # user typed its number), "ocr" (confident marker read), "spatial" (order
    # guess — the note says verify, so NOT training-grade), "" (blank)
    sources = ["typed" if n == 1 and assigned[i] else
               ("spatial" if assigned[i] else "") for i in range(n)]
    note = ""
    valid = [i for i in range(n) if len(polys[i]) >= 3]
    if (etiquetas and len(etiquetas) == len(valid) and len(valid) > 1
            and assign_etiquetas is not None):
        try:
            sub = assign_etiquetas(
                [[(p.x, p.y) for p in polys[i]] for i in valid], etiquetas
            )
            assigned = [""] * n
            for i, a in zip(valid, sub):
                assigned[i] = a
                sources[i] = "spatial" if a else ""
            note = " Parcel numbers assigned by spatial order — verify on the map."
        except Exception:
            pass
        samples = _bundled_glyph_samples()
        if samples and match_etiquetas is not None and array is not None:
            try:
                refined, confident = _ocr_match_polygons(
                    [polys[i] for i in valid], etiquetas, geotransform, array, samples
                )
                if any(confident):
                    merged = _merge_assignments(
                        [assigned[i] for i in valid], refined, confident, etiquetas
                    )
                    for i, a, c in zip(valid, merged, confident):
                        assigned[i] = a
                        sources[i] = "ocr" if (c and a) else ("spatial" if a else "")
                    note = (f" Parcel numbers: {sum(confident)}/{len(valid)} confirmed by "
                            "marker OCR, rest by spatial order — verify on the map.")
            except Exception:
                pass
    elif not etiquetas and valid:
        # nothing typed: read each polygon's circled number straight off the
        # raster (bundled glyph library; confident reads only, blanks otherwise)
        samples = _bundled_glyph_samples()
        if samples and read_etiquetas is not None and array is not None:
            try:
                labels, confident, sugg = _ocr_read_polygons(
                    [polys[i] for i in valid], geotransform, array, samples
                )
                read_count = 0
                for i, a, c, s in zip(valid, labels, confident, sugg):
                    if c and a:
                        assigned[i] = a
                        sources[i] = "ocr"
                        read_count += 1
                    elif s:
                        suggestions[i] = s
                if read_count:
                    note = (f" {read_count}/{len(valid)} parcel number(s) read from "
                            "the circled markers — verify on the map.")
            except Exception:
                pass
    out = dict(result)
    out["assigned_etiquetas"] = assigned
    out["etiqueta_suggestions"] = suggestions
    out["etiqueta_sources"] = sources
    out["etiqueta_note"] = note
    return out


class ParcelGeometryPlugin:
    """QGIS plugin entry point: toolbar action + ROI tools + extraction worker.

    The user selects a georeferenced TIFF raster layer, draws a polygon ROI
    over the parcel(s), and the plugin runs ``extract_with_cv2`` on a
    background thread and adds one memory layer holding the detected parcels.
    """

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.translator = install_translator(self.plugin_dir)
        self.action = None
        self.review_action = None
        self.publish_action = None
        self.toolbar = None
        self.tool = None
        self.previous_tool = None
        self._marker_pick_tool = None
        self._marker_pick_previous_tool = None
        self.settings = None
        self._fill_dialog = None
        self._publish_dialog = None
        self.worker = None
        self._worker_path = None
        self._worker_layer = None
        self._worker_crs = None
        self._worker_settings = None
        self._worker_geotransform = None

    def initGui(self):
        self.action = QAction(
            QIcon(os.path.join(self.plugin_dir, "icon.svg")),
            tr("Extract parcel(s)"),
            self.iface.mainWindow(),
        )
        self.action.setCheckable(True)
        self.action.setToolTip(tr("Draw a polygon around one parcel/group and extract it."))
        self.action.triggered.connect(self.run)
        self.review_action = QAction(
            QIcon(os.path.join(self.plugin_dir, "icon_review.svg")),
            tr("Review extracted geometry"),
            self.iface.mainWindow(),
        )
        self.review_action.setToolTip(
            tr("Open or resume review of one selected temporary parcel or block layer.")
        )
        self.review_action.triggered.connect(self.run_review)
        self.publish_action = QAction(
            QIcon(os.path.join(self.plugin_dir, "icon_publish.svg")),
            tr("Save extracted geometry to selected layer"),
            self.iface.mainWindow(),
        )
        self.publish_action.setToolTip(
            tr("Append extracted parcels or blocks to the matching selected project layer.")
        )
        self.publish_action.triggered.connect(self.run_publish)
        self.toolbar = self.iface.addToolBar("Relex Geoplan")
        self.toolbar.setObjectName("ParcelGeometryToolbar")
        self.toolbar.addAction(self.action)
        self.toolbar.addAction(self.review_action)
        self.toolbar.addAction(self.publish_action)
        self.iface.addPluginToRasterMenu("&Relex Geoplan", self.action)
        self.iface.addPluginToRasterMenu("&Relex Geoplan", self.review_action)
        self.iface.addPluginToRasterMenu("&Relex Geoplan", self.publish_action)

    def unload(self):
        for action in (self.action, self.review_action, self.publish_action):
            if action is not None:
                self.iface.removePluginRasterMenu("&Relex Geoplan", action)
        if self.toolbar is not None:
            self.toolbar.removeAction(self.action)
            self.toolbar.removeAction(self.review_action)
            self.toolbar.removeAction(self.publish_action)
            self.toolbar.deleteLater()
            self.toolbar = None
        if self.tool is not None:
            self.tool.deactivate()
        self._cancel_marker_pick()
        if self._fill_dialog is not None:
            self._fill_dialog.close()
        if self._publish_dialog is not None:
            self._publish_dialog.close()
        if self.translator is not None:
            QCoreApplication.removeTranslator(self.translator)
            self.translator = None

    def run(self, checked=False):
        """Toolbar handler: validate the selection, then arm the ROI map tool.

        Checks dependencies and that a single GDAL-readable raster layer is
        selected, shows the settings dialog, then activates the polygon ROI
        tool whose ``polygonFinished`` signal drives extraction. A second
        click while a tool is active cancels it (toggle behaviour).
        """
        if self.tool is not None:
            self.cancel_rectangle_tool("Selection cancelled.")
            return

        selected = self._selected_raster()
        if selected is None:
            return
        layer, path = selected

        plano_meta = load_sidecar(path) or parse_filename(os.path.basename(path))
        dialog = ExtractionSettingsDialog(self.iface.mainWindow(), plano_meta=plano_meta)
        if dialog.exec() != QDialog.Accepted:
            self.action.setChecked(False)
            return
        self.settings = dialog.values()
        save_sidecar(path, self.settings.get("plano_meta") or {})

        self.iface.messageBar().pushInfo(
            "Relex Geoplan",
            tr("Click to add polygon vertices. Right-click to finish. "
               "Right-click with fewer than 3 vertices, or press Esc, to cancel."),
        )
        self.previous_tool = self.iface.mapCanvas().mapTool()
        self.tool = PolygonRoiTool(self.iface.mapCanvas())
        self.tool.polygonFinished.connect(
            lambda points: self.extract_from_polygon(layer, path, points)
        )
        self.tool.cancelled.connect(
            lambda: self.cancel_rectangle_tool("Polygon selection cancelled.")
        )
        self.iface.mapCanvas().setMapTool(self.tool)
        self.action.setChecked(True)

    def _selected_raster(self):
        """Common validation: deps present + exactly one GDAL-readable raster selected.
        Returns (layer, path) or None (after showing the appropriate message)."""
        if _MISSING_DEPENDENCY is not None:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Relex Geoplan dependencies missing",
                _dependency_install_message(_MISSING_DEPENDENCY),
            )
            return None
        selected = self.iface.layerTreeView().selectedLayers()
        if len(selected) != 1 or selected[0].type() != QgsMapLayer.RasterLayer:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Relex Geoplan",
                "Select exactly one raster layer: the georeferenced TIFF.",
            )
            return None
        layer = selected[0]
        path = layer.dataProvider().dataSourceUri().split("|")[0]
        try:
            ds = gdal.Open(path)
        except RuntimeError:
            ds = None
        if ds is None:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Relex Geoplan",
                "The selected raster is not a local GDAL-readable TIFF.",
            )
            return None
        return layer, path

    def cancel_rectangle_tool(self, message=None):
        canvas = self.iface.mapCanvas()
        tool = self.tool
        if self.tool is not None:
            self.tool.deactivate()
            self.tool = None

        if tool is not None and canvas.mapTool() == tool:
            canvas.unsetMapTool(tool)

        if self.previous_tool is not None and self.previous_tool != tool:
            canvas.setMapTool(self.previous_tool)
            self.previous_tool = None
        else:
            self.previous_tool = None
            self.iface.actionPan().trigger()

        if self.action is not None:
            self.action.setChecked(False)
        if message:
            self.iface.messageBar().pushInfo("Relex Geoplan", message)

    def _canvas_to_layer_transform(self, layer):
        """Transform from the map-canvas (project) CRS to the raster layer's CRS.

        The map tools return coordinates in the canvas/project CRS, but the geotransform
        maps world->pixel in the *raster's* CRS. When the two differ, canvas coordinates
        must be reprojected first, or the ROI lands in the wrong place.
        """
        canvas_crs = self.iface.mapCanvas().mapSettings().destinationCrs()
        return canvas_crs, QgsCoordinateTransform(
            canvas_crs, layer.crs(), QgsProject.instance()
        )

    def _points_to_layer_crs(self, layer, points):
        try:
            canvas_crs, transform = self._canvas_to_layer_transform(layer)
            if canvas_crs == layer.crs():
                return points
            return [transform.transform(point) for point in points]
        except Exception:
            return points

    def extract_from_polygon(self, layer, path, world_points):
        """Build an ROI from the drawn polygon and start the worker.

        The polygon's bounding box (expanded by a stroke-width margin) is the
        analysis ROI; the polygon itself is passed on as the clip/limit.

        Args:
            layer: The selected QgsRasterLayer (defines the raster CRS).
            path: Filesystem path to the GDAL-readable TIFF.
            world_points: Polygon vertices (QgsPointXY) in the map-canvas CRS.
        """
        world_points = self._points_to_layer_crs(layer, world_points)
        array, geotransform, _crs_wkt = read_tiff(path)
        points = [Point(x=p.x(), y=p.y()) for p in world_points]
        roi, _pixel_coords = roi_from_world_polygon(points, geotransform, array.shape)
        line_width_px = (self.settings or {}).get("line_width_px", 14.0)
        roi = expand_roi(roi, array.shape, int(round(line_width_px * 5)))
        if roi.row1 <= roi.row0 or roi.col1 <= roi.col0:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Relex Geoplan",
                "Selected polygon does not overlap the raster.",
            )
            self.cancel_rectangle_tool()
            return

        self._start_worker(layer, path, array, geotransform, roi, polygon_world=points)

    def _ring_to_geom(self, ring):
        pts = [QgsPointXY(p.x, p.y) for p in ring]
        if pts and pts[0] != pts[-1]:
            pts.append(pts[0])
        geom = QgsGeometry.fromPolygonXY([pts])
        return geom if not geom.isEmpty() and geom.area() > 0 else None

    def _start_worker(self, layer, path, array, geotransform, roi, polygon_world=None):
        settings = self.settings or {}
        self._worker_layer = layer
        # CRS is a value type — copy it now so the async callback never dereferences the
        # raster layer (which may be None or a deleted C++ wrapper by the time it fires).
        self._worker_crs = layer.crs()
        self._worker_settings = settings
        self._worker_geotransform = geotransform
        self._worker_path = path
        self.iface.messageBar().pushInfo("Relex Geoplan", "Extracting parcel lines…")

        self.worker = ExtractionWorker(
            array, geotransform, roi, settings, polygon_world
        )
        self.worker.resultReady.connect(self._on_extraction_finished)
        self.worker.failed.connect(self._on_extraction_error)
        self.worker.start()

    def _on_extraction_finished(self, result):
        """Worker callback: build the output layers, reporting any failure
        instead of dying silently inside the Qt slot."""
        try:
            self._handle_extraction_result(result)
        except Exception:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Relex Geoplan",
                f"Failed to build the output layers:\n{traceback.format_exc()}",
            )
            self.worker = None
            self._worker_layer = None
            self._worker_crs = None
            self._worker_settings = None
            self._worker_geotransform = None
            self._worker_path = None
            self.cancel_rectangle_tool()

    def _handle_extraction_result(self, result):
        """Turn the extracted geometry into QGIS layers: one memory layer with
        all parcel polygons (or a line layer when no polygons were produced)."""
        settings = self._worker_settings or {}
        crs = self._worker_crs
        geotransform = self._worker_geotransform
        if crs is None or geotransform is None:
            return
        crs_id = crs.authid()
        layer_name = settings.get("layer_name", "Parcel line candidates")
        world_lines = result["polylines"]
        output_polygons = "polygons" in result
        display_line_width_mm = 0.5

        plano_meta = settings.get("plano_meta") or {}
        etiquetas = settings.get("etiquetas") or []
        output_layers = []
        suggestions_by_id = {}
        if output_polygons and settings.get("manzana_mode"):
            layer, msg = self._build_manzana_layer(
                result, plano_meta, crs_id, layer_name
            )
            if layer is not None:
                output_layers.append(layer)
        elif output_polygons:
            # assignment (spatial order + marker OCR) is computed in the worker
            # thread; assigned_etiquetas aligns with result["polygons"]
            spatial_note = result.get("etiqueta_note", "")
            assigned_full = result.get("assigned_etiquetas") or [
                etiquetas[i] if i < len(etiquetas) else ""
                for i in range(len(result["polygons"]))
            ]
            suggestions_full = result.get("etiqueta_suggestions") or [""] * len(
                result["polygons"]
            )
            sources_full = result.get("etiqueta_sources") or [""] * len(
                result["polygons"]
            )
            rings = []
            for orig_idx, ring in enumerate(result["polygons"]):
                if len(ring) < 3:  # rings are stored open; a triangle has 3 points
                    continue
                geom = self._ring_to_geom(ring)
                if geom is not None:
                    rings.append((geom, len(ring), orig_idx))

            vector_layer = QgsVectorLayer(
                f"Polygon?crs={crs_id}"
                f"&field=id:integer"
                f"&field=area_m2:double"
                f"&field=vertices:integer"
                f"&field=cca:string(50)"
                f"&field=etiqueta:string(50)"
                f"&field=sec:string(3)"
                f"&field=mz:string(4)"
                f"&field=reviewed:integer"
                f"&field=etiqueta_suggestion:string(50)"
                f"&field=nmp:string(50)"
                f"&field=fecha:string(50)",
                f"[Pc] {layer_name}",
                "memory",
            )
            vector_layer.setCustomProperty(
                "parcel_geometry/source_path", self._worker_path or ""
            )
            vector_layer.setCustomProperty("parcel_geometry/role", "source")
            vector_layer.setCustomProperty("parcel_geometry/kind", "parcela")
            vector_layer.setCustomProperty(
                "parcel_geometry/harvest_enabled", bool(settings.get("harvest_enabled"))
            )
            provider = vector_layer.dataProvider()
            features = []
            total_area = 0.0
            harvest_rings, harvest_confirms = [], []
            for idx, (geom, vertices, orig_idx) in enumerate(rings):
                raw = assigned_full[orig_idx] if orig_idx < len(assigned_full) else ""
                etiqueta = normalize_etiqueta(raw) if raw else ""
                sec = str(plano_meta.get("sec", "")).strip()
                mz = str(plano_meta.get("mz", "")).strip()
                feature_meta = dict(plano_meta, sec=sec, mz=mz)
                prefix = build_codigo(feature_meta)
                cca = build_cca(feature_meta, etiqueta) if etiqueta else prefix
                etiqueta_source = (
                    sources_full[orig_idx] if orig_idx < len(sources_full) else ""
                )
                if not etiqueta:
                    suggestions_by_id[idx] = suggestions_full[orig_idx]
                if etiqueta and etiqueta_source in ("typed", "ocr"):
                    # flywheel: training-grade labels only (spatial-order guesses
                    # carry a "verify on the map" caveat and are excluded)
                    harvest_rings.append(result["polygons"][orig_idx])
                    harvest_confirms.append(
                        (len(harvest_rings) - 1, etiqueta, etiqueta_source)
                    )
                nmp = plano_meta.get("nmp", "")
                fecha = plano_meta.get("fecha", "")
                feature = QgsFeature()
                feature.setFields(vector_layer.fields())
                feature.setGeometry(geom)
                feature.setAttributes(
                    [
                        idx, geom.area(), vertices, cca, etiqueta, sec, mz, 0,
                        suggestions_full[orig_idx], nmp, fecha,
                    ]
                )
                features.append(feature)
                total_area += geom.area()
            self._harvest_markers(
                harvest_rings, harvest_confirms, plano_meta,
                enabled=bool(settings.get("harvest_enabled")),
            )
            if features:
                provider.addFeatures(features)
                vector_layer.updateExtents()
                vector_layer.setRenderer(QgsSingleSymbolRenderer(_parcel_fill_symbol()))
                output_layers.append(vector_layer)

            expected = _expected_count(settings)
            if not output_layers:
                msg = (
                    "No parcel found. The drawn boundary may be faint or broken — try "
                    "'Recover weak shared boundaries', lower the line width, or draw the "
                    "selection tighter around a single parcel."
                )
            elif expected and len(features) < expected:
                msg = (
                    f"Added {len(features)} of {expected} expected parcel(s) "
                    f"({total_area:.1f} m² total). For faint shared edges, enable "
                    "'Recover weak shared boundaries'."
                )
            else:
                msg = (f"Added {len(features)} parcel(s) in one layer "
                       f"({total_area:.1f} m² total). Review cadastral values before saving."
                       f"{spatial_note}")

            if settings.get("debug_layers"):
                self._add_debug_line_layer(world_lines, crs_id, display_line_width_mm, layer_name)
        else:
            vector_layer = QgsVectorLayer(
                f"LineString?crs={crs_id}"
                f"&field=id:integer"
                f"&field=length_m:double"
                f"&field=vertices:integer",
                layer_name,
                "memory",
            )
            provider = vector_layer.dataProvider()
            features = []
            for i, line in enumerate(world_lines):
                if len(line) < 2:
                    continue
                length_m = sum(
                    math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(line, line[1:])
                )
                feature = QgsFeature()
                feature.setFields(vector_layer.fields())
                feature.setGeometry(
                    QgsGeometry.fromPolylineXY([QgsPointXY(p.x, p.y) for p in line])
                )
                feature.setAttributes([i, length_m, len(line)])
                features.append(feature)
            provider.addFeatures(features)
            vector_layer.updateExtents()
            symbol = QgsLineSymbol.createSimple(
                {"line_width": str(display_line_width_mm), "color": "255,0,0,255"}
            )
            vector_layer.setRenderer(QgsSingleSymbolRenderer(symbol))
            output_layers.append(vector_layer)
            msg = f"Added {len(features)} candidate lines."

        if settings.get("debug_layers") and "debug" in result:
            self._load_debug_layers(result, geotransform, crs)

        for vector_layer in output_layers:
            QgsProject.instance().addMapLayer(vector_layer)

        if output_polygons and not output_layers:
            self.iface.messageBar().pushWarning("Relex Geoplan", msg)
        else:
            self.iface.messageBar().pushInfo("Relex Geoplan", msg)

        if output_polygons and output_layers and not settings.get("manzana_mode"):
            self._prompt_review_parcels(output_layers[0], plano_meta, suggestions_by_id)

        self.worker = None
        self._worker_layer = None
        self._worker_crs = None
        self._worker_settings = None
        self._worker_geotransform = None
        self._worker_path = None
        self.cancel_rectangle_tool()

    def _build_manzana_layer(self, result, plano_meta, crs_id, layer_name="manzana"):
        """Remove shared parcel edges while preserving the arrangement exterior."""
        rings = [r for r in result.get("polygons", []) if len(r) >= 3]
        geoms = [g for g in (self._ring_to_geom(ring) for ring in rings) if g is not None]
        if not geoms:
            return None, ("No block outline found. Draw the selection around "
                          "one block; a faint boundary may need a lower line width.")
        geom = QgsGeometry.unaryUnion(geoms)
        if geom.isEmpty() or geom.isMultipart():
            return None, ("The detected parcels do not form one connected block. "
                          "Adjust the selection and extract again.")
        polygon = geom.asPolygon()
        if not polygon or not polygon[0]:
            return None, "The joined block outline is invalid."
        # interior rings mean the union had holes = parcels the extraction
        # missed; the outline is still built but the user must be told
        dropped_holes = len(polygon) > 1
        geom = QgsGeometry.fromPolygonXY([polygon[0]])
        codigo = build_codigo(plano_meta)
        layer = QgsVectorLayer(
            f"Polygon?crs={crs_id}"
            f"&field=id:integer"
            f"&field=codigo:string(15)"
            f"&field=dep:string(2)"
            f"&field=mun:string(2)"
            f"&field=sec:string(3)"
            f"&field=chac:string(4)"
            f"&field=mz:string(4)"
            f"&field=nmp:string(50)",
            f"[Mz] {layer_name}",
            "memory",
        )
        layer.setCustomProperty("parcel_geometry/source_path", self._worker_path or "")
        layer.setCustomProperty("parcel_geometry/role", "source")
        layer.setCustomProperty("parcel_geometry/kind", "manzana")
        feature = QgsFeature()
        feature.setFields(layer.fields())
        feature.setGeometry(geom)
        components = []
        for key, width in CCA_COMPONENTS:
            value = str(plano_meta.get(key, "")).strip()
            components.append(
                value.zfill(width) if value.isdigit() and len(value) <= width else ""
            )
        feature.setAttributes([0, codigo, *components, plano_meta.get("nmp", "")])
        layer.dataProvider().addFeatures([feature])
        layer.updateExtents()
        layer.setRenderer(QgsSingleSymbolRenderer(_parcel_fill_symbol()))
        if codigo:
            msg = (f"Added block {codigo} outline ({geom.area():.1f} m²). Select "
                   "a blocks (manzanas) layer in the Layers panel and use the save button.")
        else:
            msg = ("Added block outline, but the cadastral designation is incomplete — "
                   "fill dep/mun/sec/chac/mz before saving (the block code is empty).")
        if dropped_holes:
            msg += (" Warning: the joined parcels left interior gaps (undetected "
                    "parcels?) that were filled — verify the outline on the map.")
        return layer, msg

    def _harvest_store_dir(self):
        """Harvest directory in the current QGIS project folder, so the training
        store lives with the project the user works in — not wherever the
        parcelas layer happens to be loaded from (which scattered the store and
        silently disabled harvesting when no/>1 parcelas layer was loaded).
        None when the project is unsaved (no project folder yet)."""
        home = QgsProject.instance().homePath()
        if home and os.path.isdir(home):
            return os.path.join(home, "etiqueta_harvest")
        return None

    def _harvest_markers(self, rings_world, confirmed, plano_meta,
                         array=None, geotransform=None, enabled=None):
        """Training flywheel: crop the marker disk of each confirmed parcel into
        the harvest store. Uses the live worker's raster by default; the
        non-modal fill dialog passes its captured (array, geotransform) instead
        because the worker is long gone by Save time. Best-effort — never
        raises, never blocks the user."""
        try:
            if enabled is None:
                enabled = bool((self._worker_settings or {}).get("harvest_enabled"))
            if not enabled or harvest_confirmed is None or not confirmed:
                return
            store = self._harvest_store_dir()
            if array is None and self.worker is not None:
                array = self.worker.array
            gt = geotransform if geotransform is not None else self._worker_geotransform
            if not store or array is None or gt is None:
                return
            prepared = _ocr_crop(rings_world, gt, array)
            if prepared is None:
                return
            crop, rel = prepared
            harvest_confirmed(store, crop, rel, confirmed,
                              (plano_meta or {}).get("nmp", ""))
        except Exception:
            pass

    def _attach_marker_thumbnails(self, blanks, raster_context=None):
        """Attach marker thumbnails from the live or reloaded extraction raster."""
        try:
            if marker_disks is None:
                return
            if raster_context is not None:
                array, gt = raster_context
            else:
                if self.worker is None:
                    return
                array, gt = self.worker.array, self._worker_geotransform
            if gt is None or array is None:
                return
            rings, idx = [], []
            for j, b in enumerate(blanks):
                ring = _geom_exterior_points(b["geom"])
                if ring:
                    rings.append(ring)
                    idx.append(j)
            if not rings:
                return
            prepared = _ocr_crop(rings, gt, array)
            if prepared is None:
                return
            crop, rel = prepared
            for j, disk in zip(idx, marker_disks(crop, rel, reader=_bundled_char_reader())):
                if disk is not None:
                    blanks[j]["pixmap"] = _disk_to_pixmap(disk)
        except Exception:
            pass

    def _cancel_marker_pick(self):
        canvas = self.iface.mapCanvas()
        tool = self._marker_pick_tool
        if tool is not None and canvas.mapTool() is tool:
            prev = self._marker_pick_previous_tool
            if prev is not None:
                canvas.setMapTool(prev)
            else:
                canvas.unsetMapTool(tool)
        self._marker_pick_tool = None
        self._marker_pick_previous_tool = None

    def _start_marker_pick(self, dialog, row_index, layer, harvest_ctx):
        """One-shot map tool for the fill dialog's Pick button. Click the number
        for a centred crop, or drag a rectangle around the marker for exact
        control. This bypasses ellipse/selection guesses for hard parcels."""
        if crop_at_rect is None or disk_at_point is None:
            return
        array, gt = harvest_ctx
        if array is None or gt is None:
            self.iface.messageBar().pushWarning(
                "Relex Geoplan", "Marker picking is unavailable after raster cleanup."
            )
            return
        self._cancel_marker_pick()
        canvas = self.iface.mapCanvas()
        tool = MarkerPickTool(canvas)
        self._marker_pick_tool = tool
        self._marker_pick_previous_tool = canvas.mapTool()
        self.iface.messageBar().pushInfo(
            "Relex Geoplan",
            "Click the number, or drag a box around the marker if clicking misses."
        )

        def to_px(x, y):
            det = gt.pixel_width * gt.pixel_height - gt.rot_x * gt.rot_y
            dx, dy = x - gt.origin_x, y - gt.origin_y
            return ((dx * gt.pixel_height - dy * gt.rot_x) / det,
                    (dy * gt.pixel_width - dx * gt.rot_y) / det)

        def point_to_layer(point):
            src = canvas.mapSettings().destinationCrs()
            try:
                dst = layer.crs() if layer is not None else src
            except RuntimeError:
                return None
            if src.isValid() and dst.isValid() and src != dst:
                return QgsCoordinateTransform(src, dst, QgsProject.instance()).transform(point)
            return point

        def finish(got, pick=None):
            if got is None:
                self.iface.messageBar().pushWarning(
                    "Relex Geoplan", "No marker crop could be made from that pick."
                )
            else:
                dialog.set_picked_disk(row_index, got[0], pick)

        def clicked(point):
            try:
                point = point_to_layer(point)
                if point is None:
                    finish(None)
                    return
                col, row = to_px(point.x(), point.y())
                finish(disk_at_point(array, col, row),
                       {"pick_px": [round(col, 1), round(row, 1)]})
            finally:
                self._cancel_marker_pick()

        def rect_picked(a, b):
            try:
                a, b = point_to_layer(a), point_to_layer(b)
                if a is None or b is None:
                    finish(None)
                    return
                c0, r0 = to_px(a.x(), a.y())
                c1, r1 = to_px(b.x(), b.y())
                finish(crop_at_rect(array, c0, r0, c1, r1),
                       {"pick_rect_px": [round(c0, 1), round(r0, 1),
                                         round(c1, 1), round(r1, 1)]})
            finally:
                self._cancel_marker_pick()

        tool.pointPicked.connect(clicked)
        tool.rectPicked.connect(rect_picked)
        tool.cancelled.connect(self._cancel_marker_pick)
        canvas.setMapTool(tool)

    def _prompt_review_parcels(
        self, layer, plano_meta, suggestions_by_id, raster_context=None
    ):
        """Review every extracted parcel before it can be saved."""
        rows = []
        for feat in layer.getFeatures():
            geom = feat.geometry()
            valid = geom is not None and not geom.isEmpty()
            parcel_id = feat["id"]
            rows.append({
                "fid": feat.id(),
                "id": parcel_id,
                "area": geom.area() if valid else 0.0,
                "geom": QgsGeometry(geom) if valid else None,
                "etiqueta": str(feat["etiqueta"] or "").strip(),
                "sec": str(feat["sec"] or "").strip(),
                "mz": str(feat["mz"] or "").strip(),
                "suggestion": (
                    suggestions_by_id.get(parcel_id, "")
                    or str(feat["etiqueta_suggestion"] or "").strip()
                ),
                "pixmap": None,
            })
        if not rows:
            return
        if raster_context is None and self.worker is not None:
            raster_context = (self.worker.array, self._worker_geotransform)
        self._attach_marker_thumbnails(rows, raster_context)
        if getattr(self, "_fill_dialog", None) is not None:
            self.iface.messageBar().pushWarning(
                "Relex Geoplan",
                "Another parcel review is open. Finish or close it, then use "
                "'Review extracted parcels' on this temporary layer.",
            )
            return
        dialog = ReviewParcelsDialog(self.iface.mainWindow(), self.iface, layer, rows)
        dialog.set_pick_callback(
            lambda row: self._start_marker_pick(dialog, row, layer, harvest_ctx)
        )
        # NON-MODAL: the numbers on the plano are often tiny, so the user must
        # keep the map tools (zoom/pan) while typing. The dialog floats above
        # QGIS (Qt.Tool) and Save applies asynchronously. Capture the raster
        # refs now — the worker is nulled right after this method returns, but
        # the harvest flywheel still needs the array at Save time.
        harvest_ctx = raster_context or (None, None)
        dialog.setWindowFlags(dialog.windowFlags() | Qt.Tool)
        dialog.setModal(False)
        dialog.accepted.connect(
            lambda: self._apply_reviewed_parcels(
                dialog, layer, plano_meta, rows, harvest_ctx
            )
        )
        dialog.finished.connect(lambda _r: (self._cancel_marker_pick(),
                                            setattr(self, "_fill_dialog", None)))
        self._fill_dialog = dialog
        dialog.show()

    def _apply_reviewed_parcels(self, dialog, layer, plano_meta, rows, harvest_ctx):
        """Apply reviewed metadata and temporary-layer deletions."""
        try:
            fields = layer.fields() if layer is not None else None
        except RuntimeError:
            return
        if fields is None:
            return
        indexes = {
            name: fields.indexOf(name)
            for name in ("etiqueta", "cca", "sec", "mz", "reviewed")
        }
        if any(indexes[name] < 0 for name in ("etiqueta", "cca", "sec", "mz")):
            return
        changes = {}
        filled = []
        for reviewed in dialog.values():
            fid = reviewed["fid"]
            text = reviewed["etiqueta"]
            et = normalize_etiqueta(text) if text else ""
            sec = reviewed["sec"].zfill(3)
            mz = reviewed["mz"].zfill(4)
            feature_meta = dict(plano_meta, sec=sec, mz=mz)
            cca = build_cca(feature_meta, et) if et else build_codigo(feature_meta)
            attrs = {
                indexes["etiqueta"]: et,
                indexes["cca"]: cca,
                indexes["sec"]: sec,
                indexes["mz"]: mz,
            }
            if indexes["reviewed"] >= 0:
                attrs[indexes["reviewed"]] = 1
            changes[fid] = attrs
            if et and reviewed["etiqueta_changed"]:
                filled.append((fid, et))
        deleted = dialog.deleted_fids()
        try:
            if deleted:
                layer.dataProvider().deleteFeatures(deleted)
            if changes:
                layer.dataProvider().changeAttributeValues(changes)
            layer.updateExtents()
            layer.triggerRepaint()
        except RuntimeError:
            return  # layer was deleted while the dialog was open
        self.iface.messageBar().pushInfo(
            "Relex Geoplan",
            f"Reviewed {len(changes)} parcel(s); deleted {len(deleted)} temporary parcel(s).",
        )
        # flywheel: a user-typed number is the strongest label — harvest it
        geom_by_fid = {row["fid"]: row["geom"] for row in rows}
        picked = dialog.picked_disks()
        picks = dialog.picked_points()
        harvest_enabled = bool(layer.customProperty(
            "parcel_geometry/harvest_enabled", False
        ))
        store = self._harvest_store_dir() if harvest_enabled else None
        nmp = (plano_meta or {}).get("nmp", "")
        rings_world, confirms = [], []
        for fid, et in filled:
            disk = picked.get(fid)
            if disk is not None:
                if store and harvest_disk is not None:
                    harvest_disk(store, disk, et, nmp, source="click",
                                 extra=picks.get(fid))
                continue
            geom = geom_by_fid.get(fid)
            ring = _geom_exterior_points(geom)
            if ring:
                rings_world.append(ring)
                confirms.append((len(rings_world) - 1, et, "user"))
        array, gt = harvest_ctx
        self._harvest_markers(
            rings_world, confirms, plano_meta, array=array, geotransform=gt,
            enabled=harvest_enabled,
        )

    def run_review(self, checked=False):
        """Open or resume review for one selected temporary extraction layer."""
        if self._fill_dialog is not None:
            self._fill_dialog.show()
            self._fill_dialog.raise_()
            self._fill_dialog.activateWindow()
            return
        selected = [
            layer for layer in self.iface.layerTreeView().selectedLayers()
            if _is_temporary_parcel_layer(layer) or _is_temporary_manzana_layer(layer)
        ]
        if len(selected) != 1:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Relex Geoplan",
                tr("Select exactly one temporary parcel or block layer to review."),
            )
            return
        layer = selected[0]
        if _is_temporary_manzana_layer(layer):
            self._prompt_review_block(layer)
            return
        features = list(layer.getFeatures())
        meta = {}
        feature = features[0] if features else None
        if feature is not None:
            prefix = str(feature["cca"] or "").strip()[:15]
            if len(prefix) == 15 and prefix.isdigit():
                meta.update({
                    "dep": prefix[0:2], "mun": prefix[2:4], "sec": prefix[4:7],
                    "chac": prefix[7:11], "mz": prefix[11:15],
                })
            meta["nmp"] = str(feature["nmp"] or "").strip()
            meta["fecha"] = str(feature["fecha"] or "").strip()
        raster_context = None
        source_path = str(
            layer.customProperty("parcel_geometry/source_path", "") or ""
        )
        if source_path and os.path.isfile(source_path):
            try:
                array, geotransform, _crs_wkt = read_tiff(source_path)
                raster_context = (array, geotransform)
            except Exception:
                pass
        self._prompt_review_parcels(layer, meta, {}, raster_context=raster_context)

    def _prompt_review_block(self, layer):
        if self._fill_dialog is not None:
            self.iface.messageBar().pushWarning(
                "Relex Geoplan",
                "Another review is open. Finish or close it, then use "
                "'Review extracted geometry' on this temporary layer.",
            )
            return
        features = list(layer.getFeatures())
        if len(features) != 1:
            QMessageBox.warning(
                self.iface.mainWindow(), "Relex Geoplan",
                tr("A temporary block layer must contain exactly one feature."),
            )
            return
        dialog = ReviewBlockDialog(self.iface.mainWindow(), features[0])
        dialog.setWindowFlags(dialog.windowFlags() | Qt.Tool)
        dialog.setModal(False)
        dialog.accepted.connect(lambda: self._apply_reviewed_block(dialog, layer))
        dialog.finished.connect(lambda _result: setattr(self, "_fill_dialog", None))
        self._fill_dialog = dialog
        dialog.show()

    def _apply_reviewed_block(self, dialog, layer):
        values = dialog.values()
        codigo = build_codigo(values)
        if not codigo:
            return
        fields = layer.fields()
        changes = {}
        for key, _width in CCA_COMPONENTS:
            index = fields.indexOf(key)
            if index >= 0:
                changes[index] = values[key]
        codigo_index = fields.indexOf("codigo")
        if codigo_index >= 0:
            changes[codigo_index] = codigo
        features = list(layer.getFeatures())
        if not features:
            return
        try:
            layer.dataProvider().changeAttributeValues({features[0].id(): changes})
            layer.triggerRepaint()
        except RuntimeError:
            return
        self.iface.messageBar().pushInfo(
            "Relex Geoplan", tr("Updated block code: {code}").format(code=codigo)
        )

    def run_publish(self, checked=False):
        """Open the NON-MODAL publish dialog. The user keeps working in QGIS
        while it is open (finish reviews, fix etiquetas, inspect layers on the
        map), so parcel records are re-collected live in the dialog and again
        at Save time — never a stale snapshot from open time."""
        if self._publish_dialog is not None:
            try:
                self._publish_dialog.close()
            except RuntimeError:
                pass
            self._publish_dialog = None
        db_layer = self._selected_destination_layer()
        if db_layer is None:
            return
        eligible = [
            layer for layer in QgsProject.instance().mapLayers().values()
            if _is_temporary_publish_layer(layer)
        ]
        if not eligible:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Relex Geoplan",
                tr("No temporary parcel or block layers found. Extract first."),
            )
            return

        preselect = set(self.iface.layerTreeView().selectedLayers())
        dialog = PublishDialog(
            self.iface.mainWindow(), eligible, preselect, db_layer,
            collect=self._collect_publish_records,
            on_save=self._apply_publish,
            highlight=self._flash_publish_layer,
            on_check=self._set_publish_selection,
            source_pred=_is_temporary_publish_layer,
            kind="geometry",
        )
        self._publish_dialog = dialog
        dialog.finished.connect(self._on_publish_dialog_finished)
        dialog.show()

    def _on_publish_dialog_finished(self, _result):
        self._publish_dialog = None

    def _apply_publish(self, db_layer, chosen):
        """Save handler of the non-modal publish dialog. ``chosen`` carries the
        freshly re-collected (layer, records) pairs. Returns True on success so
        the dialog closes; False keeps it open for the user to fix and retry."""
        records = [rec for _layer, recs in chosen for rec in recs]
        if not records:
            self.iface.messageBar().pushWarning("Relex Geoplan", tr("No features to save."))
            return False
        # blocks append plainly like parcels (user decision): duplicate codigos
        # are the user's to clean up manually in QGIS, no prompt
        count, error = self._append_to_parcelas_layer(db_layer, records)
        if error:
            self.iface.messageBar().pushWarning("Relex Geoplan", error)
            return False
        self.iface.messageBar().pushInfo(
            "Relex Geoplan",
            f"Saved {count} feature(s) to project layer '{db_layer.name()}'.",
        )
        try:
            db_layer.reload()
            db_layer.triggerRepaint()
        except RuntimeError:
            pass
        return True

    def _flash_publish_layer(self, layer):
        """'Locate' button of a publish row: activate the layer and flash its
        parcels on the canvas so the user sees exactly which layer the row
        refers to. Best-effort — never raises."""
        try:
            self.iface.setActiveLayer(layer)
            ids = layer.allFeatureIds()
            if ids:
                self.iface.mapCanvas().flashFeatureIds(layer, ids)
        except Exception:
            pass

    def _set_publish_selection(self, layer, checked):
        """Checked publish rows keep their parcels SELECTED on the canvas (the
        project's selection color), a persistent view of what will be saved.
        Cleared on uncheck and when the dialog closes. Best-effort."""
        try:
            if checked:
                layer.selectAll()
            else:
                layer.removeSelection()
        except Exception:
            pass

    def _selected_destination_layer(self):
        selected = [
            layer
            for layer in self.iface.layerTreeView().selectedLayers()
            if _is_publish_destination_layer(layer)
        ]
        if len(selected) == 1:
            return selected[0]
        QMessageBox.warning(
            self.iface.mainWindow(),
            "Relex Geoplan",
            tr("Select exactly one writable polygon layer with cca and etiqueta "
               "fields in the Layers panel."),
        )
        return None

    def _collect_manzana_records(self, layer):
        """Manzana temp layer -> publishable records. A row needs geometry and
        a complete 15-digit code; etiqueta echoes its four-digit block number."""
        records, incomplete = [], []
        for feature in layer.getFeatures():
            geom = feature.geometry()
            if geom is None or geom.isEmpty():
                continue
            meta = {
                key: str(feature[key] or "").strip()
                for key, _width in CCA_COMPONENTS
            }
            codigo = build_codigo(meta)
            if codigo:
                records.append((
                    QgsGeometry(geom),
                    {"cca": codigo, "etiqueta": codigo[11:15]},
                    layer.crs(),
                ))
            else:
                idx = feature.fields().indexOf("id")
                incomplete.append(feature["id"] if idx >= 0 else feature.id())
        return records, incomplete

    def _collect_publish_records(self, layer):
        if _is_temporary_manzana_layer(layer):
            return self._collect_manzana_records(layer)
        return self._collect_layer_records(layer)

    def _collect_layer_records(self, layer):
        """Split a temporary parcel layer into publishable records and the fids of
        incomplete parcels. A parcel is publishable only with a full cca AND a
        valid etiqueta and completed review. The CCA is rebuilt from the current
        per-feature sec/mz fields so attribute-table edits cannot leave it stale.
        A wrong or empty parcel number would corrupt the cca, so it blocks local save.
        Returns (records, incomplete_fids)."""
        records, incomplete = [], []
        for feature in layer.getFeatures():
            geom = feature.geometry()
            if geom is None or geom.isEmpty():
                continue
            raw_cca = str(feature["cca"] or "").strip()
            prefix = raw_cca[:15]
            etiqueta = normalize_etiqueta(feature["etiqueta"])
            feature_meta = {}
            if len(prefix) == 15 and prefix.isdigit():
                feature_meta = {
                    "dep": prefix[0:2], "mun": prefix[2:4],
                    "sec": str(feature["sec"] or "").strip(),
                    "chac": prefix[7:11], "mz": str(feature["mz"] or "").strip(),
                }
            cca = build_cca(feature_meta, etiqueta)
            reviewed = bool(feature["reviewed"])
            if cca and etiqueta_valid(etiqueta) and reviewed:
                records.append((
                    QgsGeometry(geom),
                    {
                        "cca": cca,
                        "etiqueta": etiqueta,
                        "nmp": feature["nmp"] or "",
                        # NOT the plano/registro date: `fecha` is the parcel row's
                        # last-update date, set at write time in _append_to_parcelas_layer.
                    },
                    layer.crs(),
                ))
            else:
                idx = feature.fields().indexOf("id")
                incomplete.append(feature["id"] if idx >= 0 else feature.id())
        return records, incomplete

    def _append_to_parcelas_layer(self, db_layer, records):
        """Append records through the selected project layer provider."""
        if not _is_publish_destination_layer(db_layer):
            return 0, "The selected destination layer is no longer available."
        fields = db_layer.fields()
        now = datetime.now().isoformat(sep=" ", timespec="seconds")
        features = []
        for geom, attrs, src_crs in records:
            if not src_crs.isValid():
                return 0, "A temporary extraction layer has no valid CRS."
            g = QgsGeometry(geom)
            attrs = dict(attrs)
            if db_layer.crs() != src_crs:
                try:
                    transform = QgsCoordinateTransform(
                        src_crs, db_layer.crs(), QgsProject.instance()
                    )
                    if g.transform(transform) != 0:
                        return 0, "CRS transform to the destination layer failed."
                except Exception:
                    return 0, "CRS transform to the destination layer failed."
            attrs.setdefault("ara", round(g.area(), 2))
            attrs.setdefault("st_length_", round(g.constGet().perimeter(), 2))
            attrs["created_at"] = now                 # when the parcel row was created
            attrs["fecha"] = now[:10]                 # last-update date (YYYY-MM-DD)
            feature = QgsFeature(fields)
            for key, value in attrs.items():
                i = fields.indexOf(key)
                if i >= 0 and value not in (None, ""):
                    feature.setAttribute(i, value)
            feature.setGeometry(g)
            features.append(feature)
        ok, _ = db_layer.dataProvider().addFeatures(features)
        if not ok:
            return 0, "Saving to the selected destination layer failed."
        db_layer.updateExtents()
        return len(features), None

    def _add_debug_line_layer(self, world_lines, crs_id, line_width_mm, base_name):
        line_layer = QgsVectorLayer(
            f"LineString?crs={crs_id}"
            f"&field=id:integer"
            f"&field=length_m:double"
            f"&field=vertices:integer",
            f"[pg lines] {base_name}",
            "memory",
        )
        provider = line_layer.dataProvider()
        features = []
        for i, line in enumerate(world_lines):
            if len(line) < 2:
                continue
            length_m = sum(
                math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(line, line[1:])
            )
            feature = QgsFeature()
            feature.setFields(line_layer.fields())
            feature.setGeometry(
                QgsGeometry.fromPolylineXY([QgsPointXY(p.x, p.y) for p in line])
            )
            feature.setAttributes([i, length_m, len(line)])
            features.append(feature)
        provider.addFeatures(features)
        line_layer.updateExtents()
        symbol = QgsLineSymbol.createSimple(
            {"line_width": str(max(line_width_mm * 0.5, 0.1)), "color": "255,128,0,255"}
        )
        line_layer.setRenderer(QgsSingleSymbolRenderer(symbol))
        QgsProject.instance().addMapLayer(line_layer)

    def _load_debug_layers(self, result, geotransform, crs):
        debug = result.get("debug", {})
        offset = debug.get("offset")
        if offset is None:
            return
        roi_gt = Geotransform(
            origin_x=geotransform.origin_x + offset.col * geotransform.pixel_width,
            pixel_width=geotransform.pixel_width,
            rot_x=geotransform.rot_x,
            origin_y=geotransform.origin_y + offset.row * geotransform.pixel_height,
            rot_y=geotransform.rot_y,
            pixel_height=geotransform.pixel_height,
        )
        crs_wkt = crs.toWkt()
        srs = osr.SpatialReference()
        srs.ImportFromWkt(crs_wkt)

        for name in ("threshold_mask", "boundary", "mask"):
            arr = debug.get(name)
            if arr is None:
                continue
            with tempfile.NamedTemporaryFile(suffix=f"_{name}.tif", delete=False) as f:
                tmp_path = f.name
            h, w = arr.shape[:2]
            driver = gdal.GetDriverByName("GTiff")
            ds = driver.Create(tmp_path, w, h, 1, gdal.GDT_Byte)
            ds.SetGeoTransform(roi_gt)
            ds.SetProjection(srs.ExportToWkt())
            ds.GetRasterBand(1).WriteArray((arr.astype("uint8") * 255))
            ds.FlushCache()
            ds = None
            rl = QgsProject.instance().addMapLayer(
                QgsRasterLayer(tmp_path, f"[pg] {name}")
            )
            if not rl.isValid():
                self.iface.messageBar().pushWarning(
                    "Relex Geoplan", f"Debug layer {name} failed to load."
                )

    def _on_extraction_error(self, msg):
        QMessageBox.critical(
            self.iface.mainWindow(),
            "Relex Geoplan",
            f"Extraction failed:\n{msg}",
        )
        self.worker = None
        self._worker_layer = None
        self._worker_crs = None
        self._worker_settings = None
        self._worker_geotransform = None
        self._worker_path = None
        self.cancel_rectangle_tool()


class ExtractionWorker(QThread):
    """Runs ``extract_with_cv2`` off the UI thread.

    Emits ``resultReady`` with the extraction dict on success, or ``failed``
    with a formatted traceback on error (e.g. OpenCV not installed).
    """

    resultReady = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self, array, geotransform, roi, settings, polygon_world=None, parent=None
    ):
        super().__init__(parent)
        self.array = array
        self.geotransform = geotransform
        self.roi = roi
        self.settings = settings
        self.polygon_world = polygon_world

    def run(self):
        try:
            clip_polygon = None
            if self.polygon_world is not None and self.settings.get(
                "clip_to_selection", True
            ):
                clip_polygon = self.polygon_world

            line_width_px = (
                None  # core auto-estimates the stroke width via granulometry
                if self.settings.get("auto_width", True)
                else self.settings.get("line_width_px", 14.0)
            )
            expected_count = _expected_count(self.settings)

            # Mode A: explicit single-parcel boundary trace (fill complex shapes the line
            # arrangement fragments). Single parcel only, so skip the count/recover machinery.
            if self.settings.get("trace_boundary", False):
                self.resultReady.emit(self._finalize(self._extract(line_width_px, clip_polygon, trace=True)))
                return

            result = self._extract(line_width_px, clip_polygon)
            if expected_count > 0:
                candidates = [result]
                should_retry = (
                    self.settings.get("recover_weak_shared_boundaries", False)
                    or len(result.get("polygons", [])) < expected_count
                )
                if should_retry:
                    for retry_width in (4.0, 5.0, 6.0):
                        if retry_width == line_width_px:
                            continue
                        candidates.append(self._extract(retry_width, clip_polygon))
                result = max(
                    candidates,
                    key=lambda candidate: _expected_count_score(
                        candidate.get("polygons", []), expected_count
                    ),
                )
                result = _with_largest_polygons(result, expected_count)
                # Mode B: a single parcel the arrangement fragmented (no face dominates) — the
                # boundary trace fills it. Auto only when count==1 (so a group is never merged)
                # and only if the traced face is clearly larger, so clean singles keep their
                # precise line-intersection corners.
                if expected_count == 1:
                    traced = self._extract(line_width_px, clip_polygon, trace=True)
                    if _largest_ring_area(traced) > _largest_ring_area(result) * 1.2:
                        result = _with_largest_polygons(traced, 1)
            self.resultReady.emit(self._finalize(result))
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _finalize(self, result):
        """Etiqueta assignment (spatial + marker OCR) — done here, in the worker
        thread, so a slow OCR pass can never freeze the QGIS UI."""
        try:
            return _assign_etiquetas_result(
                result, self.settings, self.geotransform, self.array
            )
        except Exception:
            return result

    def _extract(self, line_width_px, clip_polygon, trace=False):
        return extract_with_cv2(
            self.array,
            self.geotransform,
            self.roi,
            line_width_px=line_width_px,
            return_intermediates=self.settings.get("debug_layers", False),
            clip_polygon=clip_polygon,
            recover_weak_shared_boundaries=self.settings.get(
                "recover_weak_shared_boundaries", False
            ),
            trace_boundary=trace,
            try_alternate_detector=self.settings.get("try_alternate_detector", False),
        )


class ReviewBlockDialog(QDialog):
    """Edit the cadastral designation of one temporary block feature."""

    def __init__(self, parent, feature):
        super().__init__(parent)
        self.setWindowTitle(tr("Review extracted block"))
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(tr(
            "Confirm or correct the cadastral designation. The 15-digit block code "
            "is rebuilt when you save."
        )))
        form = QFormLayout()
        self._edits = {}
        labels = {
            "dep": tr("Department"), "mun": tr("Municipality"),
            "sec": tr("Section"), "chac": tr("Chacra"), "mz": tr("Block (Mz)"),
        }
        for key, width in CCA_COMPONENTS:
            edit = QLineEdit(str(feature[key] or "").strip())
            edit.setMaxLength(width)
            edit.setValidator(QRegularExpressionValidator(
                QRegularExpression(rf"^\d{{0,{width}}}$")
            ))
            edit.textChanged.connect(self._update_status)
            self._edits[key] = edit
            form.addRow(labels[key], edit)
        layout.addLayout(form)
        self._status = QLabel()
        layout.addWidget(self._status)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_status()

    def values(self):
        out = {}
        for (key, width), edit in zip(CCA_COMPONENTS, self._edits.values()):
            text = edit.text().strip()
            # zfill only real digits: an EMPTY field must stay empty so
            # build_codigo rejects it, not become a silent "0000"
            out[key] = text.zfill(width) if text.isdigit() else text
        return out

    def _update_status(self):
        codigo = build_codigo(self.values())
        self._status.setText(
            tr("Block code: {code}").format(code=codigo)
            if codigo else tr("Complete every field to build the block code.")
        )

    def _try_accept(self):
        if not build_codigo(self.values()):
            QMessageBox.warning(
                self, tr("Incomplete block data"),
                tr("Enter valid numeric values for every cadastral designation field."),
            )
            return
        self.accept()


class ReviewParcelsDialog(QDialog):
    """Review per-parcel cadastral values before local persistence."""

    def __init__(self, parent, iface, layer, parcels):
        super().__init__(parent)
        self._iface = iface
        self._layer = layer
        self._rows = []
        self._pick_cb = None
        self.setWindowTitle(tr("Review extracted parcels"))
        self.resize(920, min(760, 210 + 56 * len(parcels)))
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Review {len(parcels)} parcel(s). Confirm or correct parcel number, section "
            "and block. Delete/Restore affects only the temporary extraction layer."
        ))

        bulk = QHBoxLayout()
        self._select_all = QCheckBox(tr("Select all"))
        self._select_all.toggled.connect(self._set_all_selected)
        bulk.addWidget(self._select_all)
        self._bulk_sec = QLineEdit()
        self._bulk_sec.setPlaceholderText(tr("Section"))
        self._bulk_sec.setMaxLength(3)
        self._bulk_sec.setFixedWidth(75)
        self._bulk_sec.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\d{0,3}$")))
        bulk.addWidget(self._bulk_sec)
        set_sec = QPushButton(tr("Set section on selected"))
        set_sec.clicked.connect(lambda: self._apply_bulk("sec", self._bulk_sec.text(), 3))
        bulk.addWidget(set_sec)
        self._bulk_mz = QLineEdit()
        self._bulk_mz.setPlaceholderText(tr("Block"))
        self._bulk_mz.setMaxLength(4)
        self._bulk_mz.setFixedWidth(85)
        self._bulk_mz.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\d{0,4}$")))
        bulk.addWidget(self._bulk_mz)
        set_mz = QPushButton(tr("Set block on selected"))
        set_mz.clicked.connect(lambda: self._apply_bulk("mz", self._bulk_mz.text(), 4))
        bulk.addWidget(set_mz)
        bulk.addStretch()
        layout.addLayout(bulk)

        holder = QWidget()
        grid = QGridLayout(holder)
        for c, h in enumerate((
            "", tr("parcel"), tr("marker"), tr("parcel number"), tr("section"), tr("block"), "", "", "",
        )):
            if h:
                grid.addWidget(QLabel(f"<b>{h}</b>"), 0, c)
        first_edit = None
        for r, parcel in enumerate(parcels, start=1):
            selected = QCheckBox()
            grid.addWidget(selected, r, 0)
            parcel_label = QLabel(f"id {parcel['id']}  ({parcel['area']:.0f} m²)")
            grid.addWidget(parcel_label, r, 1)
            thumb = QLabel()
            thumb.setAlignment(Qt.AlignCenter)
            self._set_thumb(thumb, parcel.get("pixmap"))
            grid.addWidget(thumb, r, 2)
            edit = QLineEdit()
            edit.setMaxLength(4)
            edit.setValidator(QRegularExpressionValidator(
                QRegularExpression(r"^(\d{0,4}|\d{0,3}[A-Za-zÑñ])$")))
            edit.setText(parcel["etiqueta"])
            if parcel["suggestion"] and not parcel["etiqueta"]:
                edit.setPlaceholderText(f"suggestion: {parcel['suggestion']}")
            edit.editingFinished.connect(lambda e=edit: self._pad_etiqueta_field(e))
            edit.returnPressed.connect(self._focus_next_edit)
            grid.addWidget(edit, r, 3)
            sec_edit = QLineEdit(parcel["sec"])
            sec_edit.setMaxLength(3)
            sec_edit.setFixedWidth(65)
            sec_edit.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\d{0,3}$")))
            sec_edit.editingFinished.connect(lambda e=sec_edit: self._pad_numeric_field(e, 3))
            grid.addWidget(sec_edit, r, 4)
            mz_edit = QLineEdit(parcel["mz"])
            mz_edit.setMaxLength(4)
            mz_edit.setFixedWidth(75)
            mz_edit.setValidator(QRegularExpressionValidator(QRegularExpression(r"^\d{0,4}$")))
            mz_edit.editingFinished.connect(lambda e=mz_edit: self._pad_numeric_field(e, 4))
            grid.addWidget(mz_edit, r, 5)
            locate = QPushButton(tr("Locate"))
            locate.clicked.connect(lambda _c=False, g=parcel["geom"]: self._locate(g))
            grid.addWidget(locate, r, 6)
            pick = QPushButton(tr("Pick"))
            pick.setToolTip(tr("Click, then click the number or drag a rectangle around it."))
            pick.clicked.connect(lambda _c=False, i=r - 1: self._request_pick(i))
            grid.addWidget(pick, r, 7)
            delete = QPushButton(tr("Delete"))
            delete.clicked.connect(lambda _c=False, i=r - 1: self._toggle_delete(i))
            grid.addWidget(delete, r, 8)
            self._rows.append({
                "fid": parcel["fid"], "edit": edit, "sec": sec_edit, "mz": mz_edit,
                "thumb": thumb, "geom": parcel["geom"], "disk": None,
                "selected": selected, "delete": delete, "deleted": False,
                "controls": (edit, sec_edit, mz_edit, locate, pick, selected),
                "label": parcel_label,
                "original": (parcel["etiqueta"], parcel["sec"], parcel["mz"]),
            })
            if first_edit is None or (not first_edit.text() and edit.text()):
                first_edit = edit
        scroll = QScrollArea()
        scroll.setWidget(holder)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        if first_edit is not None:
            first_edit.setFocus()

        self._hint = QLabel("")
        self._hint.setStyleSheet("color: #cc3333;")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Cancel).setText(tr("Review later"))
        buttons.accepted.connect(self._try_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _try_accept(self):
        """Block Save until every retained row has valid cadastral values."""
        bad = []
        for row in self._rows:
            if row["deleted"]:
                continue
            edit = row["edit"]
            t = edit.text().strip()
            if t and not etiqueta_valid(t):
                edit.setStyleSheet(self._BAD_STYLE)
                bad.append(t)
            else:
                edit.setStyleSheet("")
            for key, width in (("sec", 3), ("mz", 4)):
                field = row[key]
                value = field.text().strip()
                if not value.isdigit() or len(value) > width:
                    field.setStyleSheet(self._BAD_STYLE)
                    bad.append(f"{key}={value or 'blank'}")
                else:
                    field.setText(value.zfill(width))
                    field.setStyleSheet("")
        if bad:
            self._hint.setText(
                "Fix invalid parcel number/section/block values: " + ", ".join(bad)
            )
            return
        self.accept()

    _BAD_STYLE = "QLineEdit { border: 1px solid #cc3333; }"

    def set_pick_callback(self, callback):
        self._pick_cb = callback

    def _set_all_selected(self, selected):
        for row in self._rows:
            if not row["deleted"]:
                row["selected"].setChecked(selected)

    def _apply_bulk(self, key, value, width):
        value = str(value or "").strip()
        if not value.isdigit() or len(value) > width:
            self._hint.setText(f"Enter a numeric {key} up to {width} digits first.")
            return
        applied = 0
        for row in self._rows:
            if row["selected"].isChecked() and not row["deleted"]:
                row[key].setText(value.zfill(width))
                applied += 1
        self._hint.setText(f"Updated {key} on {applied} selected parcel(s).")

    def _toggle_delete(self, row_index):
        row = self._rows[row_index]
        row["deleted"] = not row["deleted"]
        row["delete"].setText(tr("Restore") if row["deleted"] else tr("Delete"))
        row["label"].setStyleSheet("color: #888; text-decoration: line-through;" if row["deleted"] else "")
        for control in row["controls"]:
            control.setEnabled(not row["deleted"])

    def _set_thumb(self, label, pixmap):
        label.setMinimumWidth(72)
        label.setAlignment(Qt.AlignCenter)
        if pixmap is not None:
            label.setPixmap(pixmap)
            label.setText("")
        else:
            label.clear()
            label.setText("—")

    def _request_pick(self, row_index):
        if self._pick_cb is None:
            self._hint.setText("Marker picking is not available for this extraction.")
            return
        self._hint.setText("Click the number, or drag a box around the marker.")
        self._pick_cb(row_index)

    def set_picked_disk(self, row_index, disk, pick=None):
        if not (0 <= row_index < len(self._rows)) or disk is None:
            return
        row = self._rows[row_index]
        row["disk"] = disk
        row["pick"] = pick  # raster location of the user's pick (gold detector label)
        self._set_thumb(row["thumb"], _disk_to_pixmap(disk))
        row["edit"].setFocus()
        self._hint.setText("Picked marker crop for this parcel. Type its number, then Save.")

    def picked_disks(self):
        return {row["fid"]: row["disk"] for row in self._rows
                if row.get("disk") is not None}

    def picked_points(self):
        return {row["fid"]: row["pick"] for row in self._rows
                if row.get("disk") is not None and row.get("pick")}

    def _pad_etiqueta_field(self, edit):
        """On commit, show the canonical DB form the value will be stored as
        (upper-cased, left-zero-padded: '3a' -> '003A'). A value whose parcel
        number is zero ('000A', '0000') is flagged instead of padded — the
        letter is a subdivision of a numbered parcel, so it can't be zero."""
        t = edit.text().strip()
        if not t:
            edit.setStyleSheet("")
        elif etiqueta_valid(t):
            edit.setText(normalize_etiqueta(t))
            edit.setStyleSheet("")
        else:
            edit.setStyleSheet(self._BAD_STYLE)

    def _pad_numeric_field(self, edit, width):
        value = edit.text().strip()
        if value.isdigit() and len(value) <= width:
            edit.setText(value.zfill(width))
            edit.setStyleSheet("")
        else:
            edit.setStyleSheet(self._BAD_STYLE)

    def _focus_next_edit(self):
        """Enter jumps to the next empty etiqueta field (fast keyboard entry)."""
        edits = [row["edit"] for row in self._rows]
        sender = self.sender()
        start = edits.index(sender) + 1 if sender in edits else 0
        for e in edits[start:] + edits[:start]:
            if not e.text().strip():
                e.setFocus()
                return

    def _locate(self, geom):
        if geom is None or geom.isEmpty():
            return
        canvas = self._iface.mapCanvas()
        rect = geom.boundingBox()
        try:
            src = self._layer.crs()
        except RuntimeError:
            return
        dst = canvas.mapSettings().destinationCrs()
        if src.isValid() and dst.isValid() and src != dst:
            rect = QgsCoordinateTransform(
                src, dst, QgsProject.instance()
            ).transformBoundingBox(rect)
        rect.scale(3.0)
        canvas.setExtent(rect)
        canvas.refresh()
        try:
            canvas.flashGeometries([geom], src)
        except Exception:
            pass

    def values(self):
        values = []
        for row in self._rows:
            if row["deleted"]:
                continue
            etiqueta = row["edit"].text().strip()
            sec = row["sec"].text().strip()
            mz = row["mz"].text().strip()
            original = tuple(str(value or "").strip() for value in row["original"])
            current = (etiqueta, sec, mz)
            values.append({
                "fid": row["fid"], "etiqueta": etiqueta, "sec": sec, "mz": mz,
                "changed": current != original,
                "etiqueta_changed": etiqueta != original[0],
            })
        return values

    def deleted_fids(self):
        return [row["fid"] for row in self._rows if row["deleted"]]


class PublishDialog(QDialog):
    """Confirm reviewed temporary parcels and their selected project destination.

    NON-MODAL: the dialog stays open while the user keeps working in QGIS
    (finishing reviews, fixing etiquetas), so parcel records are re-collected
    whenever the dialog regains focus and again at Save time — the ready ✓ /
    incomplete ✗ counts are live, never a snapshot from open time. Each row's
    "Locate" button flashes that layer's parcels on the canvas, so which layer a
    row refers to needs no name cross-referencing. The destination is fixed at
    open time (a DB write target must not follow stray layer-tree clicks)."""

    def __init__(self, parent, layers, preselect, dest_layer, collect, on_save,
                 highlight, on_check=None, source_pred=_is_temporary_parcel_layer,
                 kind="parcel"):
        super().__init__(parent)
        self._kind = kind
        self._singular = "feature"
        self._plural = "features"
        self.setWindowTitle(tr("Save extracted geometry"))
        self._collect = collect
        self._on_save = on_save
        self._highlight = highlight
        self._on_check = on_check
        self._source_pred = source_pred
        self._dest = dest_layer
        self._dest_id = dest_layer.id()
        self._dest_gone = False
        self._first_refresh = True
        self._rows = []  # {widget, cb, status, layer, id, records, incomplete}
        layout = QVBoxLayout(self)

        src_group = QGroupBox(tr("Temporary parcel and block layers to save"))
        self._src_layout = QVBoxLayout(src_group)
        for layer in layers:
            self._add_row(layer, layer in preselect)
        layout.addWidget(src_group)

        dest_group = QGroupBox(tr("Selected project destination"))
        dest_layout = QVBoxLayout(dest_group)
        dest_crs = dest_layer.crs()
        crs_txt = dest_crs.authid() if dest_crs.isValid() else "unknown"
        dest_layout.addWidget(QLabel(tr("Layer: {name}").format(name=dest_layer.name())))
        dest_layout.addWidget(QLabel(tr("Provider: {name}").format(
            name=dest_layer.dataProvider().name()
        )))
        dest_layout.addWidget(QLabel(tr(
            "CRS: {crs} (geometry reprojected to match)"
        ).format(crs=crs_txt)))
        layout.addWidget(dest_group)

        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)
        self._buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self._buttons.accepted.connect(self._try_save)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        project = QgsProject.instance()
        project.layersAdded.connect(self._layers_added)
        project.layersWillBeRemoved.connect(self._layers_removed)
        self.finished.connect(self._disconnect_project)
        self._refresh()

    def _add_row(self, layer, preselect):
        row_widget = QWidget()
        h = QHBoxLayout(row_widget)
        h.setContentsMargins(0, 0, 0, 0)
        cb = QCheckBox(layer.name())
        if self._on_check is not None:
            # connected BEFORE setChecked so preselected rows highlight on open
            cb.stateChanged.connect(
                lambda state, l=layer: self._on_check(l, bool(state)))
        cb.setChecked(bool(preselect))
        cb.stateChanged.connect(self._update_hint)
        status = QLabel("")
        show = QPushButton(tr("Locate"))
        show.setToolTip(tr("Flash this layer's geometry on the map"))
        show.clicked.connect(lambda _checked=False, l=layer: self._highlight(l))
        h.addWidget(cb)
        h.addWidget(status, 1)
        h.addWidget(show)
        self._src_layout.addWidget(row_widget)
        self._rows.append({"widget": row_widget, "cb": cb, "status": status,
                           "layer": layer, "id": layer.id(),
                           "records": [], "incomplete": []})

    def _refresh(self):
        """Re-collect records for every row — the user may have reviewed, edited
        or deleted parcels since the dialog opened. Drops rows whose layer died."""
        for row in list(self._rows):
            try:
                records, incomplete = self._collect(row["layer"])
                row["cb"].setText(row["layer"].name())
            except RuntimeError:  # layer's C++ object was deleted
                self._drop_row(row)
                continue
            row["records"], row["incomplete"] = records, incomplete
            n_ok, n_bad = len(records), len(incomplete)
            if n_bad:
                ids = ", ".join(str(i) for i in incomplete[:10])
                more = " …" if n_bad > 10 else ""
                row["status"].setText(
                    tr("{ready} ready ✓ / {bad} incomplete or unreviewed ✗ "
                       "(id {ids}{more})").format(
                        ready=n_ok, bad=n_bad, ids=ids, more=more
                    ))
            else:
                row["status"].setText(tr("{count} ready ✓").format(count=n_ok))
            if self._first_refresh and row["cb"].isChecked() and not n_ok:
                row["cb"].setChecked(False)
        self._first_refresh = False
        self._update_hint()

    def changeEvent(self, event):
        # regaining focus means the user was just working in QGIS — refresh
        if event.type() == QEvent.ActivationChange and self.isActiveWindow():
            self._refresh()
        super().changeEvent(event)

    def _layers_added(self, layers):
        known = {row["id"] for row in self._rows}
        added = False
        for layer in layers:
            if self._source_pred(layer) and layer.id() not in known:
                self._add_row(layer, False)
                added = True
        if added:
            self._refresh()

    def _layers_removed(self, layer_ids):
        ids = set(layer_ids)
        if self._dest_id in ids:
            self._dest_gone = True
        for row in [r for r in self._rows if r["id"] in ids]:
            self._drop_row(row)
        self._update_hint()

    def _drop_row(self, row):
        self._rows.remove(row)
        self._src_layout.removeWidget(row["widget"])
        row["widget"].deleteLater()

    def _disconnect_project(self, _result=None):
        project = QgsProject.instance()
        for signal, slot in ((project.layersAdded, self._layers_added),
                             (project.layersWillBeRemoved, self._layers_removed)):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        if self._on_check is not None:
            for row in self._rows:  # clear the canvas selections we created
                try:
                    self._on_check(row["layer"], False)
                except RuntimeError:
                    pass

    def _chosen_and_bad(self):
        """(chosen, bad): row dicts; bad = chosen rows with incomplete parcels."""
        chosen = [r for r in self._rows if r["cb"].isChecked()]
        return chosen, [r for r in chosen if r["incomplete"]]

    def _update_hint(self):
        self._buttons.button(QDialogButtonBox.Save).setEnabled(not self._dest_gone)
        if self._dest_gone:
            self._hint.setText(
                tr("The destination layer was removed from the project. Close this "
                   "dialog and reopen it with a destination layer selected."))
            return
        chosen, bad = self._chosen_and_bad()
        if not self._rows:
            self._hint.setText(tr("No temporary parcel or block layers left. Extract first."))
        elif not chosen:
            self._hint.setText(tr("Select at least one temporary layer to save."))
        elif bad:
            details = "; ".join(
                tr("{layer}: feature(s) {ids}").format(
                    layer=r["cb"].text(),
                    ids=", ".join(str(i) for i in r["incomplete"][:12]),
                )
                for r in bad
            )
            requirement = tr("Complete the required cadastral identity and review fields.")
            template = tr("Incomplete geometry: {details}. {requirement}")
            self._hint.setText(template.format(details=details, requirement=requirement))
        else:
            total = sum(len(r["records"]) for r in chosen)
            template = tr("{count} feature(s) ready to save. Counts refresh whenever you "
                          "return to this window.")
            self._hint.setText(template.format(count=total))

    def _try_save(self):
        """Save = re-collect first (the dialog is non-modal, layers may have
        changed), then enforce the all-complete rule on the fresh counts."""
        self._refresh()
        if self._dest_gone:
            return
        chosen, bad = self._chosen_and_bad()
        if not chosen:
            QMessageBox.information(
                self,
                tr("Save geometry"),
                tr("Select at least one layer."),
            )
            return
        if bad:
            detail = "\n".join(
                tr("  • {layer}: feature id {ids}").format(
                    layer=r["cb"].text(),
                    ids=", ".join(str(i) for i in r["incomplete"]),
                )
                for r in bad
            )
            QMessageBox.warning(
                self,
                tr("Incomplete geometry data"),
                tr("These features cannot be saved:\n\n{detail}\n\nComplete their required "
                   "identity fields, then try again.").format(detail=detail),
            )
            return
        if self._on_save(self._dest, [(r["layer"], r["records"]) for r in chosen]):
            self.accept()


class ExtractionSettingsDialog(QDialog):
    """Modal dialog for extraction options, persisted via ``QSettings``.

    Collects the per-plano DB attributes, the output layer name, optional
    manual line width (when auto-detect is off), and extraction/debug toggles.
    ``values()`` returns the chosen settings and saves them for next time.
    """

    SETTINGS_PREFIX = "ParcelGeometryPlugin/"

    def __init__(self, parent=None, plano_meta=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Relex Geoplan Extraction"))
        self._settings = QSettings()
        for stale_key in ("cca_prefix_override", "save_output", "output_polygons",
                          "save_to_db", "parcelas_layer_id", "manzana_mode"):
            self._settings.remove(self.SETTINGS_PREFIX + stale_key)
        meta = plano_meta or {}

        # Per-plano attributes (pre-filled from sidecar/filename, saved back to sidecar).
        self.meta_nmp = QLineEdit(meta.get("nmp", ""))
        self.meta_nmp.setToolTip(tr("Survey plan number from the registry stamp."))
        self.meta_fecha = QLineEdit(meta.get("fecha", ""))
        self.meta_fecha.setPlaceholderText("YYYY-MM-DD")
        self.meta_parts = {}
        nomenclatura = QHBoxLayout()
        for key, label, width in (
            ("dep", "Dep", 2), ("mun", "Mun", 2), ("sec", "Sec", 3),
            ("chac", "Chac", 4), ("mz", "Mz", 4),
        ):
            edit = QLineEdit(meta.get(key, ""))
            edit.setPlaceholderText(label)
            edit.setMaxLength(width)
            edit.setFixedWidth(52)
            self.meta_parts[key] = edit
            nomenclatura.addWidget(edit)
        nomenclatura.addStretch()
        self.etiquetas = QLineEdit()
        self.etiquetas.setPlaceholderText(tr("e.g. 18 or 320E (comma-separated if several)"))
        self.etiquetas.setToolTip(
            tr("Parcel number(s) — the number inside the circle on the survey plan. Completes the "
               "cca. With several parcels, list one per expected polygon in size order.")
        )
        # Default the layer name to the plano's identity (nmp + nomenclatura) so
        # each extraction gets a distinct, recognizable layer; fall back to the
        # remembered generic name only when no metadata is available.
        generated = _default_layer_name(meta)
        self.layer_name = QLineEdit(
            generated if generated != "Parcel line candidates"
            else self._settings.value(self.SETTINGS_PREFIX + "layer_name",
                                      "Parcel line candidates")
        )

        self.line_width = QDoubleSpinBox()
        self.line_width.setRange(1.0, 80.0)
        self.line_width.setDecimals(1)
        self.line_width.setValue(
            float(self._settings.value(self.SETTINGS_PREFIX + "line_width_px", 14.0))
        )
        self.line_width.setSuffix(" px")

        self.expected_parcel_count = QSpinBox()
        self.expected_parcel_count.setRange(0, 100)
        self.expected_parcel_count.setValue(
            int(float(self._settings.value(self.SETTINGS_PREFIX + "expected_parcel_count", 0)))
        )
        self.expected_parcel_count.setSpecialValueText(tr("Unknown"))

        def _bool_setting(key, default):
            val = self._settings.value(self.SETTINGS_PREFIX + key, default)
            return val in (True, "true", "True", "1", 1)

        self.debug_layers = QCheckBox(tr("Load debug layers (masks and lines)"))
        self.debug_layers.setChecked(_bool_setting("debug_layers", False))
        self.clip_to_selection = QCheckBox(tr("Keep only output inside selection polygon"))
        self.clip_to_selection.setChecked(_bool_setting("clip_to_selection", True))
        self.use_expected_parcel_count = QCheckBox(tr("Use expected parcel count"))
        self.use_expected_parcel_count.setChecked(
            _bool_setting("use_expected_parcel_count", False)
        )
        self.use_expected_parcel_count.setToolTip(
            tr("How many parcels you selected. Keeps the N largest faces and, if fewer are found, "
               "auto-tries weak-boundary recovery. Set to 1 for a single parcel.")
        )
        self.use_expected_parcel_count.toggled.connect(self.expected_parcel_count.setEnabled)
        self.expected_parcel_count.setEnabled(self.use_expected_parcel_count.isChecked())
        self.recover_weak_shared_boundaries = QCheckBox(tr("Recover weak shared boundaries"))
        self.recover_weak_shared_boundaries.setChecked(
            _bool_setting("recover_weak_shared_boundaries", False)
        )
        self.recover_weak_shared_boundaries.setToolTip(
            tr("Use when adjacent parcels merge because a shared boundary is faint or dashed. "
               "Re-adds those weak lines. May over-split noisy drawings, so leave off unless needed.")
        )
        self.trace_boundary = QCheckBox(tr("Trace boundary (single complex parcel)"))
        self.trace_boundary.setChecked(_bool_setting("trace_boundary", False))
        self.trace_boundary.setToolTip(
            tr("For ONE large/irregular parcel that comes out fragmented: traces and fills its "
               "outline instead of building faces from lines. Single parcel only — it merges a group.")
        )
        self.try_alternate_detector = QCheckBox(tr("Try alternate detector rescue"))
        self.try_alternate_detector.setChecked(_bool_setting("try_alternate_detector", False))
        self.try_alternate_detector.setToolTip(
            tr("Advanced: if the Hough result is weak, also try OpenCV FastLineDetector candidates "
               "and keep them only when they cover the selected ROI much better.")
        )
        self.manzana_mode = QCheckBox(tr("Extract block outline (one polygon)"))
        self.manzana_mode.setChecked(False)
        self.manzana_mode.setToolTip(
            tr("Run the normal parcel extraction, then remove shared internal parcel "
               "edges to keep the exact exterior of ONE block. Identity comes from the "
               "cadastral designation below (code through block, no parcel number); save "
               "it to a blocks layer with the save button.")
        )
        self.harvest_enabled = QCheckBox(tr(
            "Save confirmed parcel-number samples for local recognition training"
        ))
        self.harvest_enabled.setChecked(_bool_setting("harvest_enabled", False))
        self.harvest_enabled.setToolTip(tr(
            "Stores confirmed marker crops in the current QGIS project folder. "
            "Samples remain local and are never uploaded."
        ))
        self.harvest_enabled.toggled.connect(self._confirm_harvest_first_use)
        self.manzana_mode.toggled.connect(self._update_mode_controls)
        self.auto_width = QCheckBox(tr("Auto-detect line width (recommended)"))
        self.auto_width.setChecked(_bool_setting("auto_width", True))
        self.auto_width.toggled.connect(
            lambda checked: self.line_width.setEnabled(not checked)
        )
        self.line_width.setEnabled(not self.auto_width.isChecked())

        form = QFormLayout()
        form.addRow(tr("Output layer name"), self.layer_name)
        form.addRow("", self.auto_width)
        form.addRow(tr("Parcel line width"), self.line_width)
        form.addRow("", self.use_expected_parcel_count)
        form.addRow(tr("Expected parcel count"), self.expected_parcel_count)
        form.addRow("", self.clip_to_selection)
        form.addRow("", self.recover_weak_shared_boundaries)
        form.addRow("", self.trace_boundary)
        form.addRow("", self.try_alternate_detector)
        form.addRow("", self.harvest_enabled)
        form.addRow("", self.manzana_mode)
        form.addRow("", self.debug_layers)

        meta_form = QFormLayout()
        meta_form.addRow(tr("Survey plan number (nmp)"), self.meta_nmp)
        meta_form.addRow(tr("Cadastral designation"), nomenclatura)
        meta_form.addRow(tr("Date"), self.meta_fecha)
        meta_form.addRow(tr("Parcel number(s)"), self.etiquetas)
        self.meta_status = QLabel()
        self.meta_status.setTextFormat(Qt.RichText)
        meta_form.addRow(tr("Status"), self.meta_status)
        meta_group = QGroupBox(tr("Cadastral attributes (database)"))
        meta_group.setLayout(meta_form)

        # live readiness: nmp / cca-prefix / fecha as ✓ or ✗, and whether
        # publish would be blocked — surfaced BEFORE extraction, not after.
        for w in (self.meta_nmp, self.meta_fecha, *self.meta_parts.values()):
            w.textChanged.connect(self._update_meta_status)
        self._update_meta_status()
        self._update_mode_controls(self.manzana_mode.isChecked())

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(meta_group)
        layout.addWidget(buttons)

    _META_LABELS = {"dep": "Dep", "mun": "Mun", "sec": "Sec",
                    "chac": "Chac", "mz": "Mz"}

    def _update_meta_status(self):
        """Live ✓/✗ for the DB attributes, with the publish-blocking one called
        out: the cca prefix (nomenclatura through manzana) must be complete or
        no parcel can get a full cca."""
        m = {"nmp": self.meta_nmp.text().strip(),
             "fecha": self.meta_fecha.text().strip(),
             **{k: e.text().strip() for k, e in self.meta_parts.items()}}
        prefix = build_codigo(m)

        def row(ok, label, detail):
            mark, color = ("✓", "#1e9e5a") if ok else ("✗", "#cc3333")
            return f"<span style='color:{color}'>{mark}</span> <b>{label}</b> — {detail}"

        lines = [row(bool(m["nmp"]), "nmp", m["nmp"] or "missing")]
        if prefix:
            nomen = "-".join(m[k].zfill(w) for k, w in CCA_COMPONENTS)
            lines.append(row(True, "cca prefix", f"{nomen} (through block)"))
        else:
            missing = [self._META_LABELS[k] for k, w in CCA_COMPONENTS
                       if not (m[k].isdigit() and 0 < len(m[k]) <= w)]
            lines.append(row(False, "cca prefix",
                             "incomplete: " + ", ".join(missing)))
        lines.append(row(bool(m["fecha"]), "date", m["fecha"] or "missing"))
        if prefix:
            if self.manzana_mode.isChecked():
                lines.append("<i style='color:#1e9e5a'>Block code is publish-ready.</i>")
            else:
                lines.append("<i style='color:#1e9e5a'>Publish-ready once each "
                             "parcel has a parcel number.</i>")
        else:
            lines.append("<i style='color:#cc3333'>Publish is blocked until the "
                         "cca prefix is complete.</i>")
        self.meta_status.setText("<br>".join(lines))

    def _update_mode_controls(self, manzana_mode):
        """Keep parcel-only controls consistent with the selected output schema."""
        for widget in (self.etiquetas, self.use_expected_parcel_count,
                       self.trace_boundary):
            widget.setEnabled(not manzana_mode)
        self.expected_parcel_count.setEnabled(
            not manzana_mode and self.use_expected_parcel_count.isChecked()
        )
        if manzana_mode:
            self.harvest_enabled.setChecked(False)
        self.harvest_enabled.setEnabled(not manzana_mode)
        self._update_meta_status()

    def _confirm_harvest_first_use(self, checked):
        if not checked or self._settings.value(
            self.SETTINGS_PREFIX + "harvest_consent_seen", False, type=bool
        ):
            return
        answer = QMessageBox.question(
            self,
            tr("Enable local training collection?"),
            tr("Confirmed parcel-number marker crops will be saved in the current "
               "QGIS project folder. They remain on this computer and are never "
               "uploaded. Disabling this option stops future collection but does not "
               "delete existing samples. Enable local collection?"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self._settings.setValue(
                self.SETTINGS_PREFIX + "harvest_consent_seen", True
            )
            return
        self.harvest_enabled.blockSignals(True)
        self.harvest_enabled.setChecked(False)
        self.harvest_enabled.blockSignals(False)

    def values(self):
        values = {
            "layer_name": self.layer_name.text().strip() or "Parcel line candidates",
            "auto_width": self.auto_width.isChecked(),
            "line_width_px": self.line_width.value(),
            "use_expected_parcel_count": self.use_expected_parcel_count.isChecked(),
            "expected_parcel_count": self.expected_parcel_count.value(),
            "debug_layers": self.debug_layers.isChecked(),
            "clip_to_selection": self.clip_to_selection.isChecked(),
            "recover_weak_shared_boundaries": self.recover_weak_shared_boundaries.isChecked(),
            "trace_boundary": self.trace_boundary.isChecked(),
            "try_alternate_detector": self.try_alternate_detector.isChecked(),
            "harvest_enabled": self.harvest_enabled.isChecked(),
            "manzana_mode": self.manzana_mode.isChecked(),
            # per-raster values: persisted in the tiff sidecar, not QSettings
            "plano_meta": {
                "nmp": self.meta_nmp.text().strip(),
                "fecha": self.meta_fecha.text().strip(),
                **{k: e.text().strip() for k, e in self.meta_parts.items()},
            },
            "etiquetas": [
                e.strip() for e in self.etiquetas.text().split(",") if e.strip()
            ],
        }
        for key, val in values.items():
            if key == "manzana_mode":
                continue
            if key == "harvest_enabled" and values["manzana_mode"]:
                # manzana mode force-unchecks the harvest checkbox in the UI;
                # that transient state must not overwrite the saved preference
                continue
            if isinstance(val, (dict, list)):
                continue
            if isinstance(val, bool):
                self._settings.setValue(self.SETTINGS_PREFIX + key, val)
            elif isinstance(val, (int, float)):
                self._settings.setValue(self.SETTINGS_PREFIX + key, float(val))
            elif hasattr(val, "value"):
                self._settings.setValue(self.SETTINGS_PREFIX + key, val.value)
            else:
                self._settings.setValue(self.SETTINGS_PREFIX + key, str(val))
        self._settings.sync()
        if values["manzana_mode"]:
            values["trace_boundary"] = False
            values["use_expected_parcel_count"] = False
            values["expected_parcel_count"] = 0
            values["etiquetas"] = []
            values["harvest_enabled"] = False
        return values


class MarkerPickTool(QgsMapTool):
    """One-shot marker picker: click for a centered crop, or drag a rectangle
    around the marker when point-picking is unreliable."""

    pointPicked = pyqtSignal(object)
    rectPicked = pyqtSignal(object, object)
    cancelled = pyqtSignal()

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self._start_screen = None
        self._start_map = None
        self.rubber_band = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self.rubber_band.setColor(QColor(255, 204, 51, 55))
        self.rubber_band.setStrokeColor(QColor(255, 204, 51, 220))
        self.rubber_band.setWidth(2)

    def canvasPressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.cancelled.emit()
            return
        if event.button() != Qt.LeftButton:
            return
        self._start_screen = event.pos()
        self._start_map = self.toMapCoordinates(event.pos())

    def canvasMoveEvent(self, event):
        if self._start_map is None or self._start_screen is None:
            return
        dx = abs(event.pos().x() - self._start_screen.x())
        dy = abs(event.pos().y() - self._start_screen.y())
        if dx + dy < 8:
            return
        self._update_rect(self._start_map, self.toMapCoordinates(event.pos()))

    def canvasReleaseEvent(self, event):
        if event.button() != Qt.LeftButton or self._start_map is None:
            return
        end = self.toMapCoordinates(event.pos())
        dx = abs(event.pos().x() - self._start_screen.x()) if self._start_screen else 0
        dy = abs(event.pos().y() - self._start_screen.y()) if self._start_screen else 0
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        start = self._start_map
        self._start_screen = None
        self._start_map = None
        if dx + dy < 8:
            self.pointPicked.emit(end)
        else:
            self.rectPicked.emit(start, end)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()

    def deactivate(self):
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        super().deactivate()

    def _update_rect(self, a, b):
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        self.rubber_band.addPoint(a, False)
        self.rubber_band.addPoint(QgsPointXY(b.x(), a.y()), False)
        self.rubber_band.addPoint(b, False)
        self.rubber_band.addPoint(QgsPointXY(a.x(), b.y()), False)
        self.rubber_band.closePoints(True)
        self.rubber_band.show()


class PolygonRoiTool(QgsMapTool):
    """Map tool to click out a polygon ROI vertex by vertex.

    Left-click adds a vertex, Backspace removes the last, and right-click
    (with >= 3 vertices) finishes and emits
    ``polygonFinished(list[QgsPointXY])``; Esc emits ``cancelled``.
    """

    polygonFinished = pyqtSignal(list)
    cancelled = pyqtSignal()

    def __init__(self, canvas):
        super().__init__(canvas)
        self.canvas = canvas
        self.vertices = []
        self.rubber_band = QgsRubberBand(canvas, QgsWkbTypes.PolygonGeometry)
        self.rubber_band.setColor(QColor(255, 0, 0, 80))
        self.rubber_band.setStrokeColor(QColor(255, 0, 0, 200))
        self.rubber_band.setWidth(2)
        self.temp_band = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
        self.temp_band.setColor(QColor(255, 0, 0, 150))
        self.temp_band.setWidth(1.5)
        self.temp_band.setLineStyle(Qt.DashLine)

    def canvasPressEvent(self, event):
        if event.button() == Qt.RightButton:
            if len(self.vertices) >= 3:
                self._finish()
            else:
                self.cancelled.emit()
            return
        if event.button() != Qt.LeftButton:
            return
        point = self.toMapCoordinates(event.pos())
        self.vertices.append(point)
        self._update_rubber_band()

    def canvasMoveEvent(self, event):
        if not self.vertices:
            return
        current = self.toMapCoordinates(event.pos())
        self.temp_band.reset(QgsWkbTypes.LineGeometry)
        self.temp_band.addPoint(self.vertices[-1], False)
        self.temp_band.addPoint(current, False)
        self.temp_band.show()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
        elif event.key() == Qt.Key_Backspace and self.vertices:
            self.vertices.pop()
            self._update_rubber_band()
            if not self.vertices:
                self.temp_band.reset(QgsWkbTypes.LineGeometry)

    def deactivate(self):
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        self.temp_band.reset(QgsWkbTypes.LineGeometry)
        super().deactivate()

    def _finish(self):
        points = list(self.vertices)
        self.vertices = []
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        self.temp_band.reset(QgsWkbTypes.LineGeometry)
        self.polygonFinished.emit(points)

    def _update_rubber_band(self):
        self.rubber_band.reset(QgsWkbTypes.PolygonGeometry)
        for point in self.vertices:
            self.rubber_band.addPoint(point, False)
        self.rubber_band.closePoints(True)
        self.rubber_band.show()
