# src/spinner.py
from pathlib import Path
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QMovie
from PySide6.QtCore import Qt, QSize

import logging

logging.basicConfig(level=logging.INFO)

class LoadingSpinner(QLabel):
    """
    QLabel-based overlay spinner. Looks for a spinner.gif under:
      - <project>/res/spinner.gif (preferred)
      - <project>/src/spinner.gif (fallback)
    If no valid GIF is found a simple "Loading..." text is shown instead.
    """
    def __init__(self, parent=None, size=64, gif_path: str | None = None):
        super().__init__(parent)
        # allow transparent background so it looks like an overlay
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAlignment(Qt.AlignCenter)

        # Determine gif path
        gif_file = None
        if gif_path:
            p = Path(gif_path)
            if p.exists():
                gif_file = p
        else:
            # __file__ will place us in .../<project>/_internal/src/spinner.py
            spinner_path = Path(__file__).resolve().parent.parent.joinpath("res", "spinner.gif")
            if spinner_path.exists():
                gif_file = spinner_path
                logging.info(f"Found spinner GIF at {spinner_path}.")
            else:
                logging.warning(f"Spinner GIF not found at {spinner_path}.")

        self._has_movie = False
        if gif_file is not None:
            movie = QMovie(str(gif_file.resolve()))
            # scale the gif frames to requested size
            movie.setScaledSize(QSize(size, size))
            if movie.isValid():
                self.movie = movie
                self.setMovie(self.movie)
                self._has_movie = True
            else:
                # invalid movie file -> fallback to text
                self.movie = None
                self.setText("Loading...")
                logging.error(f"Invalid spinner GIF.\n {movie.lastError()}")
        else:
            # no gif file -> fallback to text
            self.movie = None
            self.setText("Loading...")
            logging.warning("No spinner GIF found. Using text fallback.")

        self.hide()

    def start(self):
        """Show and start animation (if available)."""
        if self._has_movie and self.movie is not None:
            self.show()
            self.movie.start()
        else:
            # no movie -> show the text label so user still has feedback
            self.show()

    def stop(self):
        """Stop animation and hide spinner."""
        if self._has_movie and self.movie is not None:
            try:
                self.movie.stop()
            except Exception:
                pass
        self.hide()
