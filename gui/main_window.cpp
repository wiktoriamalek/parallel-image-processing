#include "main_window.hpp"
#include "image_io.hpp"
#include "image_view.hpp"
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStatusBar>
#include <QTableWidget>
#include <QVBoxLayout>
#include <QtConcurrent>
#include <algorithm>
#include <exception>
#include <initializer_list>
#include <thread>
#include <utility>

namespace {
const char* algorithmNames[] = {"Progowanie", "Otsu", "Bernsen", "Konwolucja 3×3", "Filtr medianowy 3×3",
                                "Korekcja dystorsji"};

QImage toQImage(const Image& image) {
    return QImage(image.pixels.data(), image.width, image.height, image.width, QImage::Format_Grayscale8).copy();
}

QString milliseconds(double microseconds) {
    return QString::number(microseconds / 1000.0, 'f', 3);
}

QString difference(const Image& reference, const Image& image) {
    if (reference.width != image.width || reference.height != image.height) return "inny rozmiar";
    std::size_t different = 0;
    for (std::size_t i = 0; i < reference.pixels.size(); ++i) different += reference.pixels[i] != image.pixels[i];
    if (different == 0) return "identyczny";
    return QString("%1 px różnych (%2%)").arg(different).arg(100.0 * different / reference.pixels.size(), 0, 'f', 3);
}

QWidget* parameterPage(std::initializer_list<std::pair<QString, QWidget*>> rows) {
    auto* page = new QWidget;
    auto* layout = new QFormLayout(page);
    layout->setContentsMargins(0, 0, 0, 0);
    for (const auto& [label, widget] : rows) {
        if (label.isEmpty()) layout->addRow(widget);
        else layout->addRow(label, widget);
    }
    return page;
}

QLabel* note(const QString& text) {
    auto* label = new QLabel(text);
    label->setWordWrap(true);
    label->setStyleSheet("color: gray");
    return label;
}
}

MainWindow::MainWindow() {
    setWindowTitle("Przetwarzanie obrazów: CPU i GPU");

    inputView = new ImageView;
    outputView = new ImageView;
    inputCaption = new QLabel("Obraz wejściowy");
    outputCaption = new QLabel("Wynik");
    auto* imagesLayout = new QGridLayout;
    imagesLayout->addWidget(inputCaption, 0, 0);
    imagesLayout->addWidget(outputCaption, 0, 1);
    imagesLayout->addWidget(inputView, 1, 0);
    imagesLayout->addWidget(outputView, 1, 1);
    imagesLayout->setRowStretch(1, 1);

    table = new QTableWidget(0, 5);
    table->setHorizontalHeaderLabels({"Tryb", "Czas całkowity [ms]", "Same kernele GPU [ms]",
                                      "Przyspieszenie względem CPU 1", "Wynik względem CPU 1"});
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table->verticalHeader()->hide();
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setFixedHeight(130);
    connect(table, &QTableWidget::currentCellChanged, this, [this](int row) { showResult(row); });

    auto* rightLayout = new QVBoxLayout;
    rightLayout->addLayout(imagesLayout, 1);
    rightLayout->addWidget(table);

    auto* central = new QWidget;
    auto* layout = new QHBoxLayout(central);
    layout->addWidget(createControls());
    layout->addLayout(rightLayout, 1);
    setCentralWidget(central);

    connect(&watcher, &QFutureWatcher<RunOutcome>::finished, this, &MainWindow::runFinished);
    statusBar()->showMessage("Otwórz obraz, żeby zacząć");
}

QWidget* MainWindow::createControls() {
    Parameters defaults;

    openButton = new QPushButton("Otwórz obraz…");
    connect(openButton, &QPushButton::clicked, this, &MainWindow::chooseImage);

    algorithmBox = new QComboBox;
    for (const char* name : algorithmNames) algorithmBox->addItem(name);

    thresholdSpin = new QSpinBox;
    thresholdSpin->setRange(0, 255);
    thresholdSpin->setValue(defaults.threshold);
    radiusSpin = new QSpinBox;
    radiusSpin->setRange(1, 50);
    radiusSpin->setValue(defaults.radius);
    contrastSpin = new QSpinBox;
    contrastSpin->setRange(0, 255);
    contrastSpin->setValue(defaults.minContrast);
    filterBox = new QComboBox;
    for (const auto& name : convolutionFilterNames()) filterBox->addItem(QString::fromStdString(name));
    sortingNetworkCheck = new QCheckBox("Sieć sortująca zamiast nth_element");
    sortingNetworkCheck->setChecked(defaults.sortingNetwork);
    k1Spin = new QDoubleSpinBox;
    k2Spin = new QDoubleSpinBox;
    for (auto* spin : {k1Spin, k2Spin}) {
        spin->setRange(-1.0, 1.0);
        spin->setDecimals(3);
        spin->setSingleStep(0.01);
    }
    k1Spin->setValue(defaults.k1);
    k2Spin->setValue(defaults.k2);

    parameterPages = new QStackedWidget;
    parameterPages->addWidget(parameterPage({{"Próg", thresholdSpin}}));
    parameterPages->addWidget(parameterPage({{"", note("Próg jest wyznaczany z histogramu, bez parametrów.")}}));
    parameterPages->addWidget(parameterPage({{"Promień okna", radiusSpin}, {"Minimalny kontrast", contrastSpin}}));
    parameterPages->addWidget(parameterPage({{"Filtr", filterBox}}));
    parameterPages->addWidget(parameterPage({{"", sortingNetworkCheck}}));
    parameterPages->addWidget(parameterPage({{"k1", k1Spin}, {"k2", k2Spin},
        {"", note("Wczytaj zniekształcony obraz, np. z images/mitsubishi_distorted.")}}));
    connect(algorithmBox, &QComboBox::currentIndexChanged, parameterPages, &QStackedWidget::setCurrentIndex);

    auto* algorithmGroup = new QGroupBox("Algorytm");
    auto* algorithmLayout = new QVBoxLayout(algorithmGroup);
    algorithmLayout->addWidget(algorithmBox);
    algorithmLayout->addWidget(parameterPages);

    cpu1Check = new QCheckBox("CPU, 1 wątek");
    cpuNCheck = new QCheckBox("CPU, wątków:");
    gpuCheck = new QCheckBox("GPU (OpenCL)");
    for (auto* check : {cpu1Check, cpuNCheck, gpuCheck}) check->setChecked(true);
    threadsSpin = new QSpinBox;
    threadsSpin->setRange(2, 256);
    threadsSpin->setValue(std::max(2u, std::thread::hardware_concurrency()));
    repeatsSpin = new QSpinBox;
    repeatsSpin->setRange(1, 1000);
    repeatsSpin->setValue(10);

    auto* threadsRow = new QHBoxLayout;
    threadsRow->addWidget(cpuNCheck);
    threadsRow->addWidget(threadsSpin);
    auto* modesGroup = new QGroupBox("Tryby");
    auto* modesLayout = new QFormLayout(modesGroup);
    modesLayout->addRow(cpu1Check);
    modesLayout->addRow(threadsRow);
    modesLayout->addRow(gpuCheck);
    modesLayout->addRow("Powtórzenia", repeatsSpin);
    modesLayout->addRow(note("Czas to mediana z powtórzeń."));

    runButton = new QPushButton("Uruchom");
    runButton->setEnabled(false);
    connect(runButton, &QPushButton::clicked, this, &MainWindow::run);
    saveButton = new QPushButton("Zapisz wynik…");
    saveButton->setEnabled(false);
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::saveResult);

    auto* panel = new QWidget;
    panel->setFixedWidth(300);
    auto* layout = new QVBoxLayout(panel);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(openButton);
    layout->addWidget(algorithmGroup);
    layout->addWidget(modesGroup);
    layout->addWidget(runButton);
    layout->addWidget(saveButton);
    layout->addStretch();
    return panel;
}

void MainWindow::openImage(const QString& path) {
    try {
        input = loadImage(path.toStdString());
    } catch (const std::exception& error) {
        QMessageBox::critical(this, "Błąd", QString("Nie udało się wczytać %1: %2").arg(path, error.what()));
        return;
    }
    inputView->setImage(toQImage(input));
    inputCaption->setText(QString("Obraz wejściowy: %1 (%2 × %3 px)")
                              .arg(QFileInfo(path).fileName()).arg(input.width).arg(input.height));
    results.clear();
    table->setRowCount(0);
    outputView->setImage(QImage());
    outputCaption->setText("Wynik");
    runButton->setEnabled(true);
    saveButton->setEnabled(false);
    statusBar()->clearMessage();
}

void MainWindow::chooseImage() {
    QString path = QFileDialog::getOpenFileName(this, "Otwórz obraz", "images", "Obrazy (*.png *.jpg *.jpeg *.bmp)");
    if (!path.isEmpty()) openImage(path);
}

void MainWindow::saveResult() {
    int row = table->currentRow();
    if (row < 0 || row >= static_cast<int>(results.size())) return;
    QString path = QFileDialog::getSaveFileName(this, "Zapisz wynik", "results/wynik.png", "PNG (*.png)");
    if (path.isEmpty()) return;
    try {
        saveImage(path.toStdString(), results[row].output);
        statusBar()->showMessage("Zapisano " + path);
    } catch (const std::exception& error) {
        QMessageBox::critical(this, "Błąd", QString("Nie udało się zapisać %1: %2").arg(path, error.what()));
    }
}

void MainWindow::run() {
    std::vector<std::pair<Mode, QString>> modes;
    if (cpu1Check->isChecked()) modes.emplace_back(Mode::CPU1, "CPU, 1 wątek");
    if (cpuNCheck->isChecked()) modes.emplace_back(Mode::CPUN, QString("CPU, %1 wątków").arg(threadsSpin->value()));
    if (gpuCheck->isChecked()) modes.emplace_back(Mode::GPU, "GPU");
    if (modes.empty()) {
        statusBar()->showMessage("Zaznacz co najmniej jeden tryb");
        return;
    }

    setRunning(true);
    statusBar()->showMessage("Liczenie…");
    watcher.setFuture(QtConcurrent::run([image = input, algorithm = currentAlgorithm(), parameters = currentParameters(),
                                         threads = threadsSpin->value(), repeats = repeatsSpin->value(), modes] {
        RunOutcome outcome;
        try {
            for (const auto& [mode, name] : modes) {
                ModeResult result{mode, name};
                result.timing = runAlgorithm(algorithm, mode, image, parameters, threads, repeats, result.output);
                outcome.results.push_back(std::move(result));
            }
        } catch (const std::exception& error) {
            outcome.error = error.what();
        }
        return outcome;
    }));
}

void MainWindow::runFinished() {
    setRunning(false);
    RunOutcome outcome = watcher.result();
    if (!outcome.error.isEmpty()) {
        statusBar()->clearMessage();
        QMessageBox::critical(this, "Błąd", outcome.error);
        return;
    }
    results = std::move(outcome.results);

    auto reference = std::find_if(results.begin(), results.end(),
                                  [](const ModeResult& result) { return result.mode == Mode::CPU1; });
    table->setRowCount(static_cast<int>(results.size()));
    for (int row = 0; row < static_cast<int>(results.size()); ++row) {
        const ModeResult& result = results[row];
        QString kernel = result.timing.kernelUs >= 0 ? milliseconds(result.timing.kernelUs) : "–";
        QString speedup = "–";
        QString comparison = "–";
        if (reference != results.end()) {
            speedup = QString::number(reference->timing.totalUs / result.timing.totalUs, 'f', 2) + "×";
            comparison = difference(reference->output, result.output);
        }
        QStringList cells = {result.name, milliseconds(result.timing.totalUs), kernel, speedup, comparison};
        for (int column = 0; column < cells.size(); ++column) {
            table->setItem(row, column, new QTableWidgetItem(cells[column]));
        }
    }
    int last = static_cast<int>(results.size()) - 1;
    table->selectRow(last);
    showResult(last);
    saveButton->setEnabled(true);
    statusBar()->showMessage("Gotowe");
}

void MainWindow::showResult(int row) {
    if (row < 0 || row >= static_cast<int>(results.size())) return;
    outputView->setImage(toQImage(results[row].output));
    outputCaption->setText("Wynik: " + results[row].name);
}

void MainWindow::setRunning(bool running) {
    openButton->setEnabled(!running);
    runButton->setEnabled(!running);
    saveButton->setEnabled(!running && !results.empty());
}

Algorithm MainWindow::currentAlgorithm() const {
    return static_cast<Algorithm>(algorithmBox->currentIndex());
}

Parameters MainWindow::currentParameters() const {
    Parameters parameters;
    parameters.threshold = thresholdSpin->value();
    parameters.radius = radiusSpin->value();
    parameters.minContrast = contrastSpin->value();
    parameters.filter = filterBox->currentIndex();
    parameters.sortingNetwork = sortingNetworkCheck->isChecked();
    parameters.k1 = static_cast<float>(k1Spin->value());
    parameters.k2 = static_cast<float>(k2Spin->value());
    return parameters;
}
