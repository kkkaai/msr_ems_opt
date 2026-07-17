from __future__ import annotations

from pathlib import Path


def _render_window_from_figure(fig):
    rw = fig.scene.render_window
    return getattr(rw, "_vtk_obj", rw)


def save_mayavi_figure(fig, image_path: Path, mlab=None) -> None:
    image_path = Path(image_path).expanduser().resolve()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower()

    if suffix == ".eps":
        try:
            import vtk
        except Exception as exc:
            raise RuntimeError(f"VTK not available for EPS export: {exc}") from exc

        exporter = vtk.vtkGL2PSExporter()
        exporter.SetRenderWindow(_render_window_from_figure(fig))
        exporter.SetFilePrefix(str(image_path.with_suffix("")))
        exporter.SetFileFormatToEPS()
        exporter.SetSortToBSP()
        if hasattr(exporter, "CompressOff"):
            exporter.CompressOff()
        if hasattr(exporter, "DrawBackgroundOn"):
            exporter.DrawBackgroundOn()
        exporter.Write()
        if not image_path.exists():
            raise RuntimeError(f"GL2PS exporter did not create expected file: {image_path}")
        return

    if mlab is None:
        from mayavi import mlab as _mlab

        mlab = _mlab
    mlab.savefig(str(image_path))
