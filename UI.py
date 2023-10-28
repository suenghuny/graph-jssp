import sys
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.data = None

    def initUI(self):
        self.layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_inputs_tab(), "input")
        self.tabs.addTab(self.create_optimize_tab(), "optimize")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        self.setWindowTitle("SAS")
        self.resize(1500, 800)
        self.show()

    def create_inputs_tab(self):
        tab_input = QWidget()

        self.button_file = QPushButton("Open File", self)
        self.button_file.clicked.connect(self.open_file)
        self.label_file1 = QLabel("file path : ")
        self.label_file2 = QLabel()

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["Block ID", "Plate Welding", "Front-side SAW", "Turn-over", "Rear-side SAW", "Longi. Attachment", "Longi. Welding"])
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        hbox = QHBoxLayout()
        vbox = QVBoxLayout()

        hbox.addWidget(self.button_file)
        hbox.addWidget(self.label_file1)
        hbox.addWidget(self.label_file2)
        hbox.addStretch(2)
        vbox.addLayout(hbox)
        vbox.addWidget(self.table)

        tab_input.setLayout(vbox)

        return tab_input

    def open_file(self):
        file_name = QFileDialog.getOpenFileName(self)
        self.label_file2.setText(file_name[0])

        self.data = pd.read_excel(file_name[0], engine="openpyxl")

        idx = 0
        for i, row in self.data.iterrows():
            self.table.insertRow(idx)
            for j, pt in row.iteritems():
                self.table.setItem(idx, j, QTableWidgetItem(str(pt)))
            idx += 1

    def create_optimize_tab(self):
        tab_optimize = QWidget()

        # 하이퍼파라미터 입력 그룹 박스
        groupbox1 = QGroupBox('Hyper-parameters')

        self.label_opt1 = QLabel("number of samples : ")
        self.input_opt1 = QLineEdit(self)
        self.label_opt2 = QLabel("temperature : ")
        self.input_opt2 = QLineEdit(self)
        self.label_eval1 = QLabel("error in processing time : ")
        self.input_eval1 = QLineEdit(self)
        self.label_eval2 = QLabel("number of iterations : ")
        self.input_eval2 = QLineEdit(self)

        grid = QGridLayout()
        grid.addWidget(self.label_opt1, 0, 0)
        grid.addWidget(self.input_opt1, 0, 1)
        grid.addWidget(self.label_opt2, 1, 0)
        grid.addWidget(self.input_opt2, 1, 1)
        grid.addWidget(self.label_eval1, 2, 0)
        grid.addWidget(self.input_eval1, 2, 1)
        grid.addWidget(self.label_eval2, 3, 0)
        grid.addWidget(self.input_eval2, 3, 1)

        groupbox1.setLayout(grid)

        # 최적화 및 시뮬레이션 수행 그룹 박스
        groupbox2 = QGroupBox('Opitimization')

        self.button_run = QPushButton(" " * 5 + "RUN" + " " * 5, self)
        self.button_run.clicked.connect(self.run)
        self.log = QTextBrowser()
        self.log.setOpenExternalLinks(False)

        grid = QGridLayout()
        grid.addWidget(self.button_run, 0, 0)
        grid.addWidget(self.log, 1, 0)

        groupbox2.setLayout(grid)

        # 결과 확인 그룹 박스
        groupbox3 = QGroupBox('Results')

        self.table_res = QTableWidget()
        self.table_res.setColumnCount(1)
        self.table_res.setHorizontalHeaderLabels(["block ID"])
        self.table_res.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot()
        self.canvas.draw()
        self.label_res1 = QLabel("makespan : ")
        self.label_res2 = QLabel()

        grid = QGridLayout()
        grid.addWidget(self.table_res, 0, 0, 1, 2)
        grid.addWidget(self.canvas, 0, 2, 1, 2)
        grid.addWidget(self.label_res1, 1, 0)
        grid.addWidget(self.label_res2, 1, 1)

        groupbox3.setLayout(grid)

        vbox = QVBoxLayout()
        vbox.addWidget(groupbox1)
        vbox.addWidget(groupbox2)

        hbox = QHBoxLayout()
        hbox.addLayout(vbox, stretch=2)
        hbox.addWidget(groupbox3)

        tab_optimize.setLayout(hbox)

        return tab_optimize

    def run(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())