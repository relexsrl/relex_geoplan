import os

from qgis.PyQt.QtCore import QCoreApplication, QLocale, QSettings, QTranslator


def configured_language():
    settings = QSettings()
    override = settings.value("locale/overrideFlag", False)
    locale = ""
    if override in (True, "true", "True", "1", 1):
        locale = settings.value("locale/userLocale", "")
    if not locale:
        locale = QLocale.system().name()
    return str(locale or "")[:2].lower()


def install_translator(plugin_dir):
    language = configured_language()
    path = os.path.join(plugin_dir, "i18n", f"relex_geoplan_{language}.qm")
    if not os.path.isfile(path):
        return None
    translator = QTranslator()
    if not translator.load(path):
        return None
    QCoreApplication.installTranslator(translator)
    return translator


def tr(text):
    return QCoreApplication.translate("RelexGeoplan", text)
