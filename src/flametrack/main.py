import logging
import sys

from PySide6.QtWidgets import QApplication

from flametrack.gui.main_window import MainWindow

logging.basicConfig(level=logging.WARNING)


# logging.basicConfig(level=logging.DEBUG)


def main():
    # Create the application object
    app = QApplication(sys.argv)

    # Create the main window
    window = MainWindow()

    # Show the main window
    window.show()

    # Start the event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
