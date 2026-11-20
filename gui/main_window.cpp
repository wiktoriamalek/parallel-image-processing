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
#include <optional>
#include <thread>
#include <utility>

namespace {
const char* algorithmNames[] = {"Progowanie", "Negatyw", "Kontrast", "Otsu", "Multi-Otsu", "Doliny histogramu",
                                "Bernsen", "Niblack", "Sauvola", "Konwolucja 3×3", "Filtr medianowy 3×3",
                                "Korekcja dystorsji", "Zmiana rozmiaru", "Rozrost obszaru", "Canny (krawędzie)"};

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

QString thresholdsText(const Timing& timing) {
    if (!timing.thresholds) return "–";
    QStringList values;
    for (int threshold : *timing.thresholds) values << QString::number(threshold);
    QString text = values.isEmpty() ? "brak (1 klasa)" : values.join(", ");
    if (timing.smoothingIterations >= 0) text += QString(" (iteracje: %1)").arg(timing.smoothingIterations);
    return text;
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

    table = new QTableWidget(0, 6);
    table->setHorizontalHeaderLabels({"Tryb", "Czas całkowity [ms]", "Same kernele GPU [ms]",
                                      "Przyspieszenie względem CPU 1", "Wynik względem CPU 1", "Progi"});
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(5, QHeaderView::ResizeToContents);
    table->verticalHeader()->hide();
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setFixedHeight(190);
    connect(table, &QTableWidget::currentCellChanged, this, [this](int row) { showResult(row); });

    auto* rightLayout = new QVBoxLayout;
    rightLayout->addLayout(imagesLayout, 1);
    rightLayout->addWidget(table);

    auto* central = new QWidget;
    auto* layout = new QHBoxLayout(central);
    layout->addWidget(createControls());
    layout->addLayout(rightLayout, 1);
    setCentralWidget(central);

    inputView->setClickHandler([this](QPoint point) {
        if (currentAlgorithm() != Algorithm::RegionGrowing) return;
        seedXSpin->setValue(point.x());
        seedYSpin->setValue(point.y());
    });
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
    contrastFactorSpin = new QDoubleSpinBox;
    contrastFactorSpin->setRange(-5.0, 5.0);
    contrastFactorSpin->setDecimals(2);
    contrastFactorSpin->setSingleStep(0.1);
    contrastFactorSpin->setValue(defaults.contrastFactor);
    classesSpin = new QSpinBox;
    classesSpin->setRange(2, 16);
    classesSpin->setValue(defaults.classes);
    maxClassesSpin = new QSpinBox;
    maxClassesSpin->setRange(2, 16);
    maxClassesSpin->setValue(defaults.maxClasses);
    radiusSpin = new QSpinBox;
    radiusSpin->setRange(1, 50);
    radiusSpin->setValue(defaults.radius);
    contrastSpin = new QSpinBox;
    contrastSpin->setRange(0, 255);
    contrastSpin->setValue(defaults.minContrast);
    localRadiusSpin = new QSpinBox;
    localRadiusSpin->setRange(radiusSpin->minimum(), radiusSpin->maximum());
    localRadiusSpin->setValue(defaults.radius);
    localThresholdKSpin = new QDoubleSpinBox;
    localThresholdKSpin->setRange(0.0, 1.0);
    localThresholdKSpin->setSingleStep(0.05);
    localThresholdKSpin->setValue(defaults.localThresholdK);
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
    k3Spin = new QDoubleSpinBox;
    k3Spin->setRange(-1.0, 1.0);
    k3Spin->setDecimals(3);
    k3Spin->setSingleStep(0.01);
    k3Spin->setValue(defaults.k3);
    p1Spin = new QDoubleSpinBox;
    p2Spin = new QDoubleSpinBox;
    for (auto* spin : {p1Spin, p2Spin}) {
        spin->setRange(-0.2, 0.2);
        spin->setDecimals(4);
        spin->setSingleStep(0.001);
    }
    p1Spin->setValue(defaults.p1);
    p2Spin->setValue(defaults.p2);
    distortionModelBox = new QComboBox;
    distortionModelBox->addItems({"Radialny (k1, k2)", "Brown–Conrady (k1, k2, k3, p1, p2)"});
    distortionModelBox->setCurrentIndex(defaults.distortionModel);
    auto updateDistortionModel = [this](int model) {
        for (auto* spin : {k3Spin, p1Spin, p2Spin}) spin->setEnabled(model == 1);
    };
    connect(distortionModelBox, &QComboBox::currentIndexChanged, this, updateDistortionModel);
    updateDistortionModel(defaults.distortionModel);
    interpolationBox = new QComboBox;
    interpolationBox->addItems({"Najbliższy sąsiad", "Dwuliniowa"});
    interpolationBox->setCurrentIndex(defaults.bilinear ? 1 : 0);
    scaleSpin = new QDoubleSpinBox;
    scaleSpin->setRange(0.05, 4.0);
    scaleSpin->setDecimals(2);
    scaleSpin->setSingleStep(0.25);
    scaleSpin->setValue(defaults.scale);
    seedXSpin = new QSpinBox;
    seedYSpin = new QSpinBox;
    for (auto* spin : {seedXSpin, seedYSpin}) {
        spin->setRange(0, 0);
        connect(spin, &QSpinBox::valueChanged, this, &MainWindow::updateSeedMarker);
    }
    toleranceSpin = new QSpinBox;
    toleranceSpin->setRange(0, 255);
    toleranceSpin->setValue(defaults.tolerance);
    localTilesCheck = new QCheckBox("GPU: kafelki w pamięci lokalnej");
    localTilesCheck->setChecked(defaults.localTiles);
    cannyLowSpin = new QDoubleSpinBox;
    cannyHighSpin = new QDoubleSpinBox;
    for (auto* spin : {cannyLowSpin, cannyHighSpin}) {
        spin->setRange(0.0, 1500.0);
        spin->setDecimals(1);
        spin->setSingleStep(5.0);
    }
    cannyLowSpin->setValue(defaults.cannyLow);
    cannyHighSpin->setValue(defaults.cannyHigh);

    parameterPages = new QStackedWidget;
    std::vector<int> algorithmPages;
    auto addPage = [&](QWidget* page, int algorithms = 1) {
        algorithmPages.insert(algorithmPages.end(), algorithms, parameterPages->addWidget(page));
    };
    addPage(parameterPage({{"Próg", thresholdSpin}}));
    addPage(parameterPage({{"", note("Każdy piksel p zamienia się na 255 − p, bez parametrów.")}}));
    addPage(parameterPage({{"Współczynnik k", contrastFactorSpin},
        {"", note("Każdy piksel p zamienia się na (p − 128)·k + 128, obcięte do 0…255. "
                  "k > 1 zwiększa kontrast, k < 1 go zmniejsza, k < 0 odwraca jasność.")}}));
    addPage(parameterPage({{"", note("Próg jest wyznaczany z histogramu, bez parametrów.")}}));
    addPage(parameterPage({{"Liczba klas", classesSpin},
        {"", note("K klas to K − 1 progów wyznaczanych z histogramu.")}}));
    addPage(parameterPage({{"Maks. liczba klas", maxClassesSpin},
        {"", note("Histogram jest wygładzany filtrem [1 2 1]/4, aż zostanie co najwyżej K szczytów. Progi to "
                  "dna dolin między sąsiednimi szczytami, więc klas może wyjść mniej niż K.")}}));
    addPage(parameterPage({{"Promień okna", radiusSpin}, {"Minimalny kontrast", contrastSpin}}));
    addPage(parameterPage({{"Promień okna", localRadiusSpin}, {"Współczynnik k", localThresholdKSpin},
        {"", note("m i s to średnia i odchylenie standardowe jasności w oknie (2r + 1) × (2r + 1). Niblack: "
                  "T = m − k·s, Sauvola: T = m·(1 + k·(s / 128 − 1)). Piksel jaśniejszy lub równy T staje się "
                  "biały, ciemniejszy czarny. Zwykle k = 0,2 dla Niblacka i 0,2–0,5 dla Sauvoli.")}}), 2);
    addPage(parameterPage({{"Filtr", filterBox}}));
    addPage(parameterPage({{"", sortingNetworkCheck}}));
    addPage(parameterPage({{"Model", distortionModelBox}, {"k1", k1Spin}, {"k2", k2Spin},
        {"k3", k3Spin}, {"p1", p1Spin}, {"p2", p2Spin},
        {"", note("Wczytaj zniekształcony obraz, np. z images/mitsubishi_distorted. Brown–Conrady dodaje "
                  "k3·r⁶ i dystorsję styczną p1, p2, tak jak w OpenCV.")}}));
    addPage(parameterPage({{"Interpolacja", interpolationBox}, {"Skala", scaleSpin}}));
    addPage(parameterPage({{"Ziarno x", seedXSpin}, {"Ziarno y", seedYSpin}, {"Tolerancja", toleranceSpin},
        {"", localTilesCheck},
        {"", note("Kliknij obraz wejściowy, żeby wybrać ziarno. Bez kafelków GPU dokłada 1 piksel na iterację, "
                  "więc duży obszar liczy się nawet sekundy.")}}));
    addPage(parameterPage({{"Próg dolny", cannyLowSpin}, {"Próg górny", cannyHighSpin},
        {"", note("Gauss 3×3 → Sobel (moduł i kierunek) → tłumienie niemaksymalne → podwójny próg → histereza. "
                  "Progi dotyczą modułu gradientu Sobela (0…1442). Piksel powyżej progu górnego jest krawędzią, "
                  "między progami tylko wtedy, gdy łączy się z krawędzią.")}}));
    connect(algorithmBox, &QComboBox::currentIndexChanged, this, [this, algorithmPages](int index) {
        parameterPages->setCurrentIndex(algorithmPages.at(index));
    });
    connect(algorithmBox, &QComboBox::currentIndexChanged, this, &MainWindow::updateSeedMarker);

    auto* algorithmGroup = new QGroupBox("Algorytm");
    auto* algorithmLayout = new QVBoxLayout(algorithmGroup);
    algorithmLayout->addWidget(algorithmBox);
    algorithmLayout->addWidget(parameterPages);

    cpu1Check = new QCheckBox("CPU, 1 wątek");
    cpuNCheck = new QCheckBox("CPU, wątków:");
    cpuDynamicCheck = new QCheckBox("CPU, podział dynamiczny");
    gpuCheck = new QCheckBox("GPU (OpenCL)");
    gpuZeroCopyCheck = new QCheckBox("GPU (OpenCL), zero-copy");
    for (auto* check : {cpu1Check, cpuNCheck, cpuDynamicCheck, gpuCheck, gpuZeroCopyCheck}) check->setChecked(true);
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
    modesLayout->addRow(cpuDynamicCheck);
    modesLayout->addRow(gpuCheck);
    modesLayout->addRow(gpuZeroCopyCheck);
    modesLayout->addRow("Powtórzenia", repeatsSpin);
    modesLayout->addRow(note("Czas to mediana z powtórzeń. Podział dynamiczny: tyle samo wątków, ale pobierają "
                             "małe kawałki obrazu, więc rdzenie wydajne robią więcej. Zero-copy: te same kernele "
                             "bez kopiowania obrazu (wspólna pamięć CPU i GPU), dla Multi-Otsu i dolin."));
    auto updateZeroCopy = [this] { gpuZeroCopyCheck->setEnabled(supportsZeroCopy(currentAlgorithm())); };
    connect(algorithmBox, &QComboBox::currentIndexChanged, this, updateZeroCopy);
    updateZeroCopy();

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
    seedXSpin->setRange(0, input.width - 1);
    seedYSpin->setRange(0, input.height - 1);
    seedXSpin->setValue(input.width / 2);
    seedYSpin->setValue(input.height / 2);
    updateSeedMarker();
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
    Algorithm algorithm = currentAlgorithm();
    std::vector<std::pair<Mode, QString>> modes;
    if (cpu1Check->isChecked()) modes.emplace_back(Mode::CPU1, "CPU, 1 wątek");
    if (cpuNCheck->isChecked()) {
        modes.emplace_back(Mode::CPUN, QString("CPU, %1 wątków").arg(threadsSpin->value()));
    }
    if (cpuDynamicCheck->isChecked()) {
        modes.emplace_back(Mode::CPUNDynamic, QString("CPU, %1 wątków, dynamiczny").arg(threadsSpin->value()));
    }
    if (gpuCheck->isChecked()) modes.emplace_back(Mode::GPU, "GPU");
    if (gpuZeroCopyCheck->isChecked() && gpuZeroCopyCheck->isEnabled()) {
        modes.emplace_back(Mode::GPUZeroCopy, "GPU, zero-copy");
    }
    if (modes.empty()) {
        statusBar()->showMessage("Zaznacz co najmniej jeden tryb");
        return;
    }

    setRunning(true);
    statusBar()->showMessage("Liczenie…");
    watcher.setFuture(QtConcurrent::run([image = input, algorithm, parameters = currentParameters(),
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
        QStringList cells = {result.name, milliseconds(result.timing.totalUs), kernel, speedup, comparison,
                             thresholdsText(result.timing)};
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
    const ModeResult& result = results[row];
    outputView->setImage(toQImage(result.output));
    outputCaption->setText(QString("Wynik: %1 (%2 × %3 px)").arg(result.name).arg(result.output.width)
                               .arg(result.output.height));
}

void MainWindow::setRunning(bool running) {
    openButton->setEnabled(!running);
    runButton->setEnabled(!running);
    saveButton->setEnabled(!running && !results.empty());
}

void MainWindow::updateSeedMarker() {
    if (currentAlgorithm() == Algorithm::RegionGrowing && !input.pixels.empty()) {
        inputView->setMarker(QPoint(seedXSpin->value(), seedYSpin->value()));
    } else {
        inputView->setMarker(std::nullopt);
    }
}

Algorithm MainWindow::currentAlgorithm() const {
    return static_cast<Algorithm>(algorithmBox->currentIndex());
}

Parameters MainWindow::currentParameters() const {
    Parameters parameters;
    parameters.threshold = thresholdSpin->value();
    parameters.contrastFactor = static_cast<float>(contrastFactorSpin->value());
    parameters.classes = classesSpin->value();
    parameters.maxClasses = maxClassesSpin->value();
    parameters.radius = currentAlgorithm() == Algorithm::Bernsen ? radiusSpin->value() : localRadiusSpin->value();
    parameters.minContrast = contrastSpin->value();
    parameters.localThresholdK = localThresholdKSpin->value();
    parameters.filter = filterBox->currentIndex();
    parameters.sortingNetwork = sortingNetworkCheck->isChecked();
    parameters.k1 = static_cast<float>(k1Spin->value());
    parameters.k2 = static_cast<float>(k2Spin->value());
    parameters.distortionModel = distortionModelBox->currentIndex();
    parameters.k3 = static_cast<float>(k3Spin->value());
    parameters.p1 = static_cast<float>(p1Spin->value());
    parameters.p2 = static_cast<float>(p2Spin->value());
    parameters.bilinear = interpolationBox->currentIndex() == 1;
    parameters.scale = static_cast<float>(scaleSpin->value());
    parameters.seedX = seedXSpin->value();
    parameters.seedY = seedYSpin->value();
    parameters.tolerance = toleranceSpin->value();
    parameters.localTiles = localTilesCheck->isChecked();
    parameters.cannyLow = static_cast<float>(cannyLowSpin->value());
    parameters.cannyHigh = static_cast<float>(cannyHighSpin->value());
    return parameters;
}
