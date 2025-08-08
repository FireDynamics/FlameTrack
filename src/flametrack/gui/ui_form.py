# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.9.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QDate,
    QDateTime,
    QLocale,
    QMetaObject,
    QObject,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTime,
    QUrl,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QFontDatabase,
    QGradient,
    QIcon,
    QImage,
    QKeySequence,
    QLinearGradient,
    QPainter,
    QPalette,
    QPixmap,
    QRadialGradient,
    QTransform,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenuBar,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .edge_preview_canvas import EdgePreviewCanvas
from .edgeresult_canvas import EdgeResultCanvas
from .selectable_imshow_canvas import SelectableImshowCanvas


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(Qt.WindowModality.NonModal)
        MainWindow.resize(1057, 687)
        MainWindow.setAcceptDrops(True)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_7 = QGridLayout(self.centralwidget)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_dewarping = QWidget()
        self.tab_dewarping.setObjectName("tab_dewarping")
        self.gridLayout_4 = QGridLayout(self.tab_dewarping)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_12 = QVBoxLayout()
        self.verticalLayout_12.setObjectName("verticalLayout_12")
        self.button_open_folder = QPushButton(self.tab_dewarping)
        self.button_open_folder.setObjectName("button_open_folder")

        self.verticalLayout_12.addWidget(self.button_open_folder)

        self.verticalSpacer_3 = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.verticalLayout_12.addItem(self.verticalSpacer_3)

        self.label_7 = QLabel(self.tab_dewarping)
        self.label_7.setObjectName("label_7")

        self.verticalLayout_12.addWidget(self.label_7)

        self.comboBox_experiment_type = QComboBox(self.tab_dewarping)
        self.comboBox_experiment_type.addItem("")
        self.comboBox_experiment_type.addItem("")
        self.comboBox_experiment_type.setObjectName("comboBox_experiment_type")

        self.verticalLayout_12.addWidget(self.comboBox_experiment_type)

        self.label_5 = QLabel(self.tab_dewarping)
        self.label_5.setObjectName("label_5")

        self.verticalLayout_12.addWidget(self.label_5)

        self.doubleSpinBox_plate_height = QDoubleSpinBox(self.tab_dewarping)
        self.doubleSpinBox_plate_height.setObjectName("doubleSpinBox_plate_height")
        self.doubleSpinBox_plate_height.setMaximum(10000.000000000000000)

        self.verticalLayout_12.addWidget(self.doubleSpinBox_plate_height)

        self.label_6 = QLabel(self.tab_dewarping)
        self.label_6.setObjectName("label_6")

        self.verticalLayout_12.addWidget(self.label_6)

        self.doubleSpinBox_plate_width = QDoubleSpinBox(self.tab_dewarping)
        self.doubleSpinBox_plate_width.setObjectName("doubleSpinBox_plate_width")
        self.doubleSpinBox_plate_width.setMaximum(10000.000000000000000)

        self.verticalLayout_12.addWidget(self.doubleSpinBox_plate_width)

        self.label_4 = QLabel(self.tab_dewarping)
        self.label_4.setObjectName("label_4")

        self.verticalLayout_12.addWidget(self.label_4)

        self.combo_rotation = QComboBox(self.tab_dewarping)
        self.combo_rotation.addItem("")
        self.combo_rotation.addItem("")
        self.combo_rotation.addItem("")
        self.combo_rotation.addItem("")
        self.combo_rotation.setObjectName("combo_rotation")

        self.verticalLayout_12.addWidget(self.combo_rotation)

        self.horizontalLayout_5.addLayout(self.verticalLayout_12)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.plot_dewarping = SelectableImshowCanvas(self.tab_dewarping)
        self.plot_dewarping.setObjectName("plot_dewarping")
        self.plot_dewarping.setMinimumSize(QSize(100, 100))

        self.verticalLayout_5.addWidget(self.plot_dewarping)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.button_dewarp = QPushButton(self.tab_dewarping)
        self.button_dewarp.setObjectName("button_dewarp")

        self.horizontalLayout_3.addWidget(self.button_dewarp)

        self.progress_dewarping = QProgressBar(self.tab_dewarping)
        self.progress_dewarping.setObjectName("progress_dewarping")
        self.progress_dewarping.setValue(0)

        self.horizontalLayout_3.addWidget(self.progress_dewarping)

        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 5)

        self.verticalLayout_5.addLayout(self.horizontalLayout_3)

        self.verticalLayout_5.setStretch(0, 6)
        self.verticalLayout_5.setStretch(1, 1)

        self.horizontalLayout_5.addLayout(self.verticalLayout_5)

        self.verticalLayout_6.addLayout(self.horizontalLayout_5)

        self.Sliderbox = QGroupBox(self.tab_dewarping)
        self.Sliderbox.setObjectName("Sliderbox")
        self.Sliderbox.setEnabled(True)
        self.gridLayout = QGridLayout(self.Sliderbox)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_3 = QLabel(self.Sliderbox)
        self.label_3.setObjectName("label_3")

        self.verticalLayout_3.addWidget(self.label_3)

        self.label = QLabel(self.Sliderbox)
        self.label.setObjectName("label")

        self.verticalLayout_3.addWidget(self.label)

        self.label_2 = QLabel(self.Sliderbox)
        self.label_2.setObjectName("label_2")

        self.verticalLayout_3.addWidget(self.label_2)

        self.horizontalLayout.addLayout(self.verticalLayout_3)

        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.slider_frame = QSlider(self.Sliderbox)
        self.slider_frame.setObjectName("slider_frame")
        self.slider_frame.setEnabled(False)
        self.slider_frame.setMaximum(99)
        self.slider_frame.setSliderPosition(0)
        self.slider_frame.setOrientation(Qt.Orientation.Horizontal)

        self.verticalLayout_8.addWidget(self.slider_frame)

        self.slider_scale_min = QSlider(self.Sliderbox)
        self.slider_scale_min.setObjectName("slider_scale_min")
        self.slider_scale_min.setEnabled(False)
        self.slider_scale_min.setOrientation(Qt.Orientation.Horizontal)

        self.verticalLayout_8.addWidget(self.slider_scale_min)

        self.slider_scale_max = QSlider(self.Sliderbox)
        self.slider_scale_max.setObjectName("slider_scale_max")
        self.slider_scale_max.setEnabled(False)
        self.slider_scale_max.setValue(99)
        self.slider_scale_max.setOrientation(Qt.Orientation.Horizontal)

        self.verticalLayout_8.addWidget(self.slider_scale_max)

        self.horizontalLayout.addLayout(self.verticalLayout_8)

        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)

        self.verticalLayout_6.addWidget(self.Sliderbox)

        self.gridLayout_4.addLayout(self.verticalLayout_6, 1, 0, 1, 1)

        self.tabWidget.addTab(self.tab_dewarping, "")
        self.tab_edge = QWidget()
        self.tab_edge.setObjectName("tab_edge")
        self.tab_edge.setMinimumSize(QSize(0, 0))
        self.gridLayout_8 = QGridLayout(self.tab_edge)
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.plot_edge_preview = EdgePreviewCanvas(self.tab_edge)
        self.plot_edge_preview.setObjectName("plot_edge_preview")

        self.horizontalLayout_6.addWidget(self.plot_edge_preview)

        self.verticalLayout_13 = QVBoxLayout()
        self.verticalLayout_13.setObjectName("verticalLayout_13")
        self.verticalSpacer = QSpacerItem(
            20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )

        self.verticalLayout_13.addItem(self.verticalSpacer)

        self.gridLayout_6 = QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.checkBox_mulithread = QCheckBox(self.tab_edge)
        self.checkBox_mulithread.setObjectName("checkBox_mulithread")

        self.gridLayout_6.addWidget(self.checkBox_mulithread, 1, 0, 1, 1)

        self.comboBox_flame_direction = QComboBox(self.tab_edge)
        self.comboBox_flame_direction.addItem("")
        self.comboBox_flame_direction.addItem("")
        self.comboBox_flame_direction.setObjectName("comboBox_flame_direction")

        self.gridLayout_6.addWidget(self.comboBox_flame_direction, 0, 0, 1, 1)

        self.verticalLayout_13.addLayout(self.gridLayout_6)

        self.button_find_edge = QPushButton(self.tab_edge)
        self.button_find_edge.setObjectName("button_find_edge")

        self.verticalLayout_13.addWidget(self.button_find_edge)

        self.horizontalLayout_6.addLayout(self.verticalLayout_13)

        self.horizontalLayout_6.setStretch(0, 2)
        self.horizontalLayout_6.setStretch(1, 1)

        self.verticalLayout_11.addLayout(self.horizontalLayout_6)

        self.verticalLayout_14 = QVBoxLayout()
        self.verticalLayout_14.setObjectName("verticalLayout_14")
        self.progress_edge_finding_plate1 = QProgressBar(self.tab_edge)
        self.progress_edge_finding_plate1.setObjectName("progress_edge_finding_plate1")
        self.progress_edge_finding_plate1.setValue(0)

        self.verticalLayout_14.addWidget(self.progress_edge_finding_plate1)

        self.progress_edge_finding_plate2 = QProgressBar(self.tab_edge)
        self.progress_edge_finding_plate2.setObjectName("progress_edge_finding_plate2")
        self.progress_edge_finding_plate2.setValue(0)

        self.verticalLayout_14.addWidget(self.progress_edge_finding_plate2)

        self.verticalLayout_11.addLayout(self.verticalLayout_14)

        self.verticalLayout_11.setStretch(0, 4)
        self.verticalLayout_11.setStretch(1, 1)

        self.gridLayout_8.addLayout(self.verticalLayout_11, 0, 0, 1, 1)

        self.tabWidget.addTab(self.tab_edge, "")
        self.tab_analysis = QWidget()
        self.tab_analysis.setObjectName("tab_analysis")
        self.tab_analysis.setEnabled(True)
        self.gridLayout_5 = QGridLayout(self.tab_analysis)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.slider_analysis_y = QSlider(self.tab_analysis)
        self.slider_analysis_y.setObjectName("slider_analysis_y")
        self.slider_analysis_y.setOrientation(Qt.Orientation.Vertical)
        self.slider_analysis_y.setTickPosition(QSlider.TickPosition.TicksAbove)

        self.gridLayout_2.addWidget(self.slider_analysis_y, 0, 0, 1, 1)

        self.label_9 = QLabel(self.tab_analysis)
        self.label_9.setObjectName("label_9")

        self.gridLayout_2.addWidget(self.label_9, 1, 0, 1, 1)

        self.horizontalLayout_8.addLayout(self.gridLayout_2)

        self.plot_analysis = EdgeResultCanvas(self.tab_analysis)
        self.plot_analysis.setObjectName("plot_analysis")

        self.horizontalLayout_8.addWidget(self.plot_analysis)

        self.horizontalLayout_8.setStretch(0, 1)
        self.horizontalLayout_8.setStretch(1, 12)

        self.gridLayout_5.addLayout(self.horizontalLayout_8, 0, 0, 1, 1)

        self.tabWidget.addTab(self.tab_analysis, "")

        self.gridLayout_7.addWidget(self.tabWidget, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 1057, 24))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QCoreApplication.translate("MainWindow", "Flamespread Analysis Tool", None)
        )
        self.button_open_folder.setText(
            QCoreApplication.translate("MainWindow", "Open folder", None)
        )
        self.label_7.setText(
            QCoreApplication.translate("MainWindow", "Experiment Type", None)
        )
        self.comboBox_experiment_type.setItemText(
            0, QCoreApplication.translate("MainWindow", "Lateral Flame Spread", None)
        )
        self.comboBox_experiment_type.setItemText(
            1, QCoreApplication.translate("MainWindow", "Room Corner", None)
        )

        self.label_5.setText(
            QCoreApplication.translate("MainWindow", "Plate height (mm)", None)
        )
        self.label_6.setText(
            QCoreApplication.translate("MainWindow", "Plate width (mm)", None)
        )
        self.label_4.setText(
            QCoreApplication.translate("MainWindow", "Rotate image", None)
        )
        self.combo_rotation.setItemText(
            0, QCoreApplication.translate("MainWindow", "0\u00b0", None)
        )
        self.combo_rotation.setItemText(
            1, QCoreApplication.translate("MainWindow", "90\u00b0", None)
        )
        self.combo_rotation.setItemText(
            2, QCoreApplication.translate("MainWindow", "180\u00b0", None)
        )
        self.combo_rotation.setItemText(
            3, QCoreApplication.translate("MainWindow", "270\u00b0", None)
        )

        self.button_dewarp.setText(
            QCoreApplication.translate("MainWindow", "Dewarp", None)
        )
        self.Sliderbox.setTitle("")
        self.label_3.setText(QCoreApplication.translate("MainWindow", "# Frame", None))
        self.label.setText(
            QCoreApplication.translate("MainWindow", "Scale min [%]", None)
        )
        self.label_2.setText(
            QCoreApplication.translate("MainWindow", "Scale max [%]", None)
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_dewarping),
            QCoreApplication.translate("MainWindow", "Dewarping", None),
        )
        self.checkBox_mulithread.setText(
            QCoreApplication.translate("MainWindow", "Multithread", None)
        )
        self.comboBox_flame_direction.setItemText(
            0, QCoreApplication.translate("MainWindow", "Left -> Right", None)
        )
        self.comboBox_flame_direction.setItemText(
            1, QCoreApplication.translate("MainWindow", "Right -> Left", None)
        )

        self.button_find_edge.setText(
            QCoreApplication.translate("MainWindow", "Find edges", None)
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_edge),
            QCoreApplication.translate("MainWindow", "Edge recognition", None),
        )
        self.label_9.setText(QCoreApplication.translate("MainWindow", "y-Height", None))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab_analysis),
            QCoreApplication.translate("MainWindow", "Analysis", None),
        )

    # retranslateUi
