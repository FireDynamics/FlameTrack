# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.8.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMainWindow, QMenuBar,
    QProgressBar, QPushButton, QSizePolicy, QSlider,
    QSpacerItem, QStatusBar, QTabWidget, QVBoxLayout,
    QWidget)

from gui.imshow_canvas import ImshowCanvas
from gui.selectable_imshow_canvas import SelectableImshowCanvas
from gui.edgeresult_canvas import EdgeResultCanvas

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.setWindowModality(Qt.WindowModality.NonModal)
        MainWindow.resize(1057, 687)
        MainWindow.setAcceptDrops(True)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_6 = QGridLayout(self.centralwidget)
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab_dewarping = QWidget()
        self.tab_dewarping.setObjectName(u"tab_dewarping")
        self.gridLayout_4 = QGridLayout(self.tab_dewarping)
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.verticalLayout_12 = QVBoxLayout()
        self.verticalLayout_12.setObjectName(u"verticalLayout_12")
        self.button_open_folder = QPushButton(self.tab_dewarping)
        self.button_open_folder.setObjectName(u"button_open_folder")

        self.verticalLayout_12.addWidget(self.button_open_folder)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout_12.addItem(self.verticalSpacer_3)

        self.label_4 = QLabel(self.tab_dewarping)
        self.label_4.setObjectName(u"label_4")

        self.verticalLayout_12.addWidget(self.label_4)

        self.combo_rotation = QComboBox(self.tab_dewarping)
        self.combo_rotation.addItem("")
        self.combo_rotation.addItem("")
        self.combo_rotation.addItem("")
        self.combo_rotation.addItem("")
        self.combo_rotation.setObjectName(u"combo_rotation")

        self.verticalLayout_12.addWidget(self.combo_rotation)


        self.horizontalLayout_5.addLayout(self.verticalLayout_12)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.plot_dewarping = SelectableImshowCanvas(self.tab_dewarping)
        self.plot_dewarping.setObjectName(u"plot_dewarping")
        self.plot_dewarping.setMinimumSize(QSize(100, 100))

        self.verticalLayout_5.addWidget(self.plot_dewarping)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.button_dewarp = QPushButton(self.tab_dewarping)
        self.button_dewarp.setObjectName(u"button_dewarp")

        self.horizontalLayout_3.addWidget(self.button_dewarp)

        self.progress_dewarping = QProgressBar(self.tab_dewarping)
        self.progress_dewarping.setObjectName(u"progress_dewarping")
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
        self.Sliderbox.setObjectName(u"Sliderbox")
        self.Sliderbox.setEnabled(True)
        self.gridLayout = QGridLayout(self.Sliderbox)
        self.gridLayout.setObjectName(u"gridLayout")
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.label_3 = QLabel(self.Sliderbox)
        self.label_3.setObjectName(u"label_3")

        self.verticalLayout_3.addWidget(self.label_3)

        self.label = QLabel(self.Sliderbox)
        self.label.setObjectName(u"label")

        self.verticalLayout_3.addWidget(self.label)

        self.label_2 = QLabel(self.Sliderbox)
        self.label_2.setObjectName(u"label_2")

        self.verticalLayout_3.addWidget(self.label_2)


        self.horizontalLayout.addLayout(self.verticalLayout_3)

        self.verticalLayout_8 = QVBoxLayout()
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.slider_frame = QSlider(self.Sliderbox)
        self.slider_frame.setObjectName(u"slider_frame")
        self.slider_frame.setEnabled(False)
        self.slider_frame.setMaximum(99)
        self.slider_frame.setSliderPosition(0)
        self.slider_frame.setOrientation(Qt.Orientation.Horizontal)

        self.verticalLayout_8.addWidget(self.slider_frame)

        self.slider_scale_min = QSlider(self.Sliderbox)
        self.slider_scale_min.setObjectName(u"slider_scale_min")
        self.slider_scale_min.setEnabled(False)
        self.slider_scale_min.setOrientation(Qt.Orientation.Horizontal)

        self.verticalLayout_8.addWidget(self.slider_scale_min)

        self.slider_scale_max = QSlider(self.Sliderbox)
        self.slider_scale_max.setObjectName(u"slider_scale_max")
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
        self.tab_edge.setObjectName(u"tab_edge")
        self.gridLayout_3 = QGridLayout(self.tab_edge)
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.button_find_edge = QPushButton(self.tab_edge)
        self.button_find_edge.setObjectName(u"button_find_edge")

        self.horizontalLayout_7.addWidget(self.button_find_edge)

        #self.progress_edge_finding = QProgressBar(self.tab_edge)
        #self.progress_edge_finding.setObjectName(u"progress_edge_finding")
        #self.progress_edge_finding.setValue(24)

        #self.horizontalLayout_7.addWidget(self.progress_edge_finding)


        self.gridLayout_3.addLayout(self.horizontalLayout_7, 0, 0, 1, 1)

        self.tabWidget.addTab(self.tab_edge, "")
        self.tab_analysis = QWidget()
        self.tab_analysis.setObjectName(u"tab_analysis")
        self.tab_analysis.setEnabled(True)
        self.gridLayout_5 = QGridLayout(self.tab_analysis)
        self.gridLayout_5.setObjectName(u"gridLayout_5")
        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.slider_analysis_y = QSlider(self.tab_analysis)
        self.slider_analysis_y.setObjectName(u"slider_analysis_y")
        self.slider_analysis_y.setOrientation(Qt.Orientation.Vertical)
        self.slider_analysis_y.setTickPosition(QSlider.TickPosition.TicksBothSides)

        self.horizontalLayout_8.addWidget(self.slider_analysis_y)

        self.plot_analysis = EdgeResultCanvas(self.tab_analysis)
        self.plot_analysis.setObjectName(u"plot_analysis")

        self.horizontalLayout_8.addWidget(self.plot_analysis)


        self.gridLayout_5.addLayout(self.horizontalLayout_8, 0, 0, 1, 1)

        self.tabWidget.addTab(self.tab_analysis, "")

        self.gridLayout_6.addWidget(self.tabWidget, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1057, 24))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Flamespread Analysis Tool", None))
        self.button_open_folder.setText(QCoreApplication.translate("MainWindow", u"Open folder", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Rotate image", None))
        self.combo_rotation.setItemText(0, QCoreApplication.translate("MainWindow", u"0\u00b0", None))
        self.combo_rotation.setItemText(1, QCoreApplication.translate("MainWindow", u"90\u00b0", None))
        self.combo_rotation.setItemText(2, QCoreApplication.translate("MainWindow", u"180\u00b0", None))
        self.combo_rotation.setItemText(3, QCoreApplication.translate("MainWindow", u"270\u00b0", None))

        self.button_dewarp.setText(QCoreApplication.translate("MainWindow", u"Dewarp", None))
        self.Sliderbox.setTitle("")
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"# Frame", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Scale min [%]", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Scale max [%]", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_dewarping), QCoreApplication.translate("MainWindow", u"Dewarping", None))
        self.button_find_edge.setText(QCoreApplication.translate("MainWindow", u"Find edges", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_edge), QCoreApplication.translate("MainWindow", u"Edge recognition", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_analysis), QCoreApplication.translate("MainWindow", u"Analysis", None))
    # retranslateUi

