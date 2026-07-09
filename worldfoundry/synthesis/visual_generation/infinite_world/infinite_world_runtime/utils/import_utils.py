import importlib


def get_obj_from_str(string, reload=False, invalidate_cache=True):
    """
    Import an object from a fully qualified module path.

    Args:
        string: Dotted object path such as package.module.ClassName.
        reload: Reload the imported module before resolving the object.
        invalidate_cache: Invalidate importlib caches before importing.
    """
    module, cls = string.rsplit(".", 1)
    if invalidate_cache:
        importlib.invalidate_caches()
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
