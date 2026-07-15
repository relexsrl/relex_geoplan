def _add_user_site_packages():
    import site
    import sys

    user_site = site.getusersitepackages()
    if user_site not in sys.path:
        sys.path.append(user_site)


def classFactory(iface):
    _add_user_site_packages()
    from .plugin import ParcelGeometryPlugin

    return ParcelGeometryPlugin(iface)
