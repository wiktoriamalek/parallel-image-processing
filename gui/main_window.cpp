#include "main_window.hpp"
#include "gabor.hpp"
#include "image_io.hpp"
#include "image_view.hpp"
#include "median_filter.hpp"
#include "theme.hpp"
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QFrame>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QHeaderView>
#include <QLabel>
#include <QMessageBox>
#include <QPushButton>
#include <QSpinBox>
#include <QScrollArea>
#include <QSplitter>
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
class ParameterStack : public QStackedWidget {
public:
    ParameterStack() {
        layout()->setSizeConstraint(QLayout::SetNoConstraint);
        connect(this, &QStackedWidget::currentChanged, this, [this] { updateHeight(); });
    }

protected:
    void resizeEvent(QResizeEvent* event) override {
        QStackedWidget::resizeEvent(event);
        updateHeight();
    }

    bool event(QEvent* event) override {
        const bool handled = QStackedWidget::event(event);
        if (event->type() == QEvent::LayoutRequest) updateHeight();
        return handled;
    }

private:
    void updateHeight() {
        QWidget* page = currentWidget();
        if (!page) return;
        const int wrapped = page->heightForWidth(width());
        const int preferred = std::max(0, wrapped >= 0 ? wrapped : page->sizeHint().height());
        if (minimumHeight() != preferred || maximumHeight() != preferred) setFixedHeight(preferred);
    }
};

const char* algorithmNames[] = {"Progowanie", "Negatyw", "Kontrast", "Otsu", "Multi-Otsu", "Doliny histogramu",
                                "Bernsen", "Niblack", "Sauvola", "Konwolucja 3×3", "Filtr medianowy",
                                "Korekcja dystorsji", "Zmiana rozmiaru", "Rozrost obszaru", "Canny (krawędzie)",
                                "Filtr Gabora"};

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
    layout->setVerticalSpacing(8);
    layout->setRowWrapPolicy(QFormLayout::WrapAllRows);
    layout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
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
    label->setObjectName("hint");
    return label;
}
}

MainWindow::MainWindow() {
    setWindowTitle("Przetwarzanie obrazów · CPU / GPU");
    setMinimumSize(1120, 740);
    applyTheme(*this);

    inputView = new ImageView;
    inputView->setObjectName("inputPreview");
    outputView = new ImageView;
    outputView->setObjectName("outputPreview");
    inputView->setEmptyText("Wczytaj obraz do przetworzenia");
    outputView->setEmptyText("Tutaj pojawi się wynik algorytmu");
    inputCaption = new QLabel("Obraz wejściowy");
    outputCaption = new QLabel("Wynik");
    auto previewCard = [](QLabel* caption, ImageView* view) {
        auto* card = new QFrame;
        card->setObjectName("previewCard");
        caption->setObjectName("panelHeading");
        caption->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
        auto* layout = new QVBoxLayout(card);
        layout->setContentsMargins(16, 16, 16, 16);
        layout->setSpacing(14);
        layout->addWidget(caption);
        layout->addWidget(view, 1);
        return card;
    };
    auto* previews = new QSplitter(Qt::Horizontal);
    previews->setChildrenCollapsible(false);
    previews->setHandleWidth(14);
    previews->addWidget(previewCard(inputCaption, inputView));
    previews->addWidget(previewCard(outputCaption, outputView));
    previews->setStretchFactor(0, 1);
    previews->setStretchFactor(1, 1);

    table = new QTableWidget(0, 6);
    table->setHorizontalHeaderLabels({"Tryb", "Całkowity [ms]", "Etapy GPU [ms]",
                                      "Przyspieszenie", "Zgodność z CPU", "Progi"});
    table->horizontalHeaderItem(3)->setToolTip("Przyspieszenie względem CPU z jednym wątkiem");
    table->horizontalHeaderItem(4)->setToolTip("Porównanie z wynikiem CPU z jednym wątkiem");
    table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    table->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    table->horizontalHeader()->setSectionResizeMode(5, QHeaderView::Fixed);
    table->setColumnWidth(5, 110);
    table->horizontalHeader()->setMinimumSectionSize(80);
    table->horizontalHeader()->setHighlightSections(false);
    table->verticalHeader()->hide();
    table->verticalHeader()->setDefaultSectionSize(40);
    table->setSelectionBehavior(QAbstractItemView::SelectRows);
    table->setSelectionMode(QAbstractItemView::SingleSelection);
    table->setEditTriggers(QAbstractItemView::NoEditTriggers);
    table->setAlternatingRowColors(true);
    table->setShowGrid(false);
    table->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);
    table->setHorizontalScrollMode(QAbstractItemView::ScrollPerPixel);
    connect(table, &QTableWidget::currentCellChanged, this, [this](int row) { showResult(row); });

    auto* emptyResults = new QLabel("Uruchom pomiar, aby porównać wybrane tryby wykonania.");
    emptyResults->setObjectName("emptyResults");
    emptyResults->setAlignment(Qt::AlignCenter);
    emptyResults->setWordWrap(true);
    resultPages = new QStackedWidget;
    resultPages->addWidget(emptyResults);
    resultPages->addWidget(table);

    auto* resultsCard = new QFrame;
    resultsCard->setObjectName("resultsCard");
    resultsCard->setMinimumHeight(190);
    auto* resultsTitle = new QLabel("Wyniki pomiaru");
    resultsTitle->setObjectName("panelHeading");
    resultsInfo = new QLabel;
    resultsInfo->setObjectName("fileInfo");
    auto* resultsHeader = new QHBoxLayout;
    resultsHeader->addWidget(resultsTitle);
    resultsHeader->addStretch();
    resultsHeader->addWidget(resultsInfo);
    auto* resultsLayout = new QVBoxLayout(resultsCard);
    resultsLayout->setContentsMargins(16, 16, 16, 12);
    resultsLayout->setSpacing(14);
    resultsLayout->addLayout(resultsHeader);
    resultsLayout->addWidget(resultPages, 1);

    workspaceSplitter = new QSplitter(Qt::Vertical);
    workspaceSplitter->setChildrenCollapsible(false);
    workspaceSplitter->setHandleWidth(14);
    workspaceSplitter->addWidget(previews);
    workspaceSplitter->addWidget(resultsCard);
    workspaceSplitter->setStretchFactor(0, 3);
    workspaceSplitter->setStretchFactor(1, 1);
    workspaceSplitter->setSizes({520, 240});

    QWidget* controls = createControls();
    auto* title = new QLabel("Przetwarzanie obrazów");
    title->setObjectName("appTitle");
    fileInfo = new QLabel;
    fileInfo->setObjectName("fileInfo");
    fileInfo->setTextFormat(Qt::PlainText);
    fileInfo->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
    fileInfo->hide();
    auto* titleLayout = new QVBoxLayout;
    titleLayout->setContentsMargins(0, 0, 0, 0);
    titleLayout->setSpacing(5);
    titleLayout->addWidget(title);
    titleLayout->addWidget(fileInfo);
    auto* topBar = new QWidget;
    topBar->setObjectName("topBar");
    auto* topLayout = new QHBoxLayout(topBar);
    topLayout->setContentsMargins(0, 0, 0, 0);
    topLayout->setSpacing(10);
    topLayout->addLayout(titleLayout, 1);
    topLayout->addWidget(openButton);
    topLayout->addWidget(saveButton);

    auto* body = new QHBoxLayout;
    body->setSpacing(20);
    body->addWidget(controls);
    body->addWidget(workspaceSplitter, 1);
    auto* central = new QWidget;
    central->setObjectName("appRoot");
    auto* layout = new QVBoxLayout(central);
    layout->setContentsMargins(24, 20, 24, 10);
    layout->setSpacing(22);
    layout->addWidget(topBar);
    layout->addLayout(body, 1);
    setCentralWidget(central);
    statusBar()->setSizeGripEnabled(false);
    statusBar()->setContentsMargins(18, 0, 18, 0);

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

    algorithmBox = new ThemeComboBox;
    for (const char* name : algorithmNames) algorithmBox->addItem(name);

    thresholdSpin = new ThemeSpinBox;
    thresholdSpin->setRange(0, 255);
    thresholdSpin->setValue(defaults.threshold);
    contrastFactorSpin = new ThemeDoubleSpinBox;
    contrastFactorSpin->setRange(-5.0, 5.0);
    contrastFactorSpin->setDecimals(2);
    contrastFactorSpin->setSingleStep(0.1);
    contrastFactorSpin->setValue(defaults.contrastFactor);
    classesSpin = new ThemeSpinBox;
    classesSpin->setRange(2, 16);
    classesSpin->setValue(defaults.classes);
    maxClassesSpin = new ThemeSpinBox;
    maxClassesSpin->setRange(2, 16);
    maxClassesSpin->setValue(defaults.maxClasses);
    radiusSpin = new ThemeSpinBox;
    radiusSpin->setRange(1, 50);
    radiusSpin->setValue(defaults.radius);
    contrastSpin = new ThemeSpinBox;
    contrastSpin->setRange(0, 255);
    contrastSpin->setValue(defaults.minContrast);
    localRadiusSpin = new ThemeSpinBox;
    localRadiusSpin->setRange(radiusSpin->minimum(), radiusSpin->maximum());
    localRadiusSpin->setValue(defaults.radius);
    localThresholdKSpin = new ThemeDoubleSpinBox;
    localThresholdKSpin->setRange(0.0, 1.0);
    localThresholdKSpin->setSingleStep(0.05);
    localThresholdKSpin->setValue(defaults.localThresholdK);
    filterBox = new ThemeComboBox;
    for (const auto& name : convolutionFilterNames()) filterBox->addItem(QString::fromStdString(name));
    medianSortBox = new ThemeComboBox;
    medianSortBox->addItems({"Sortowanie", "Sieć, jedno okno", "Sieć bitoniczna", "Sieć współdzielona"});
    medianSortBox->setCurrentIndex(defaults.medianSort);
    medianWindowBox = new ThemeComboBox;
    for (int radius = 1; radius <= 7; ++radius) medianWindowBox->addItem(QString("%1 × %1").arg(2 * radius + 1));
    medianWindowBox->setCurrentIndex(defaults.medianRadius - 1);
    medianStripSpin = new ThemeSpinBox;
    medianStripSpin->setRange(1, 32);
    medianStripSpin->setValue(defaults.medianStrip);
    medianSwitchingCheck = new QCheckBox("Mediana przełączająca");
    medianSwitchingCheck->setChecked(defaults.medianSwitching);
    connect(medianSortBox, &QComboBox::currentIndexChanged, this, [this](int sort) {
        medianStripSpin->setEnabled(static_cast<MedianSort>(sort) == MedianSort::Shared);
    });
    connect(medianWindowBox, &QComboBox::currentIndexChanged, this, [this](int index) {
        medianStripSpin->setValue(index == 0 ? 8 : 16);
    });
    medianStripSpin->setEnabled(static_cast<MedianSort>(defaults.medianSort) == MedianSort::Shared);
    k1Spin = new ThemeDoubleSpinBox;
    k2Spin = new ThemeDoubleSpinBox;
    for (auto* spin : {k1Spin, k2Spin}) {
        spin->setRange(-1.0, 1.0);
        spin->setDecimals(3);
        spin->setSingleStep(0.01);
    }
    k1Spin->setValue(defaults.k1);
    k2Spin->setValue(defaults.k2);
    k3Spin = new ThemeDoubleSpinBox;
    k3Spin->setRange(-1.0, 1.0);
    k3Spin->setDecimals(3);
    k3Spin->setSingleStep(0.01);
    k3Spin->setValue(defaults.k3);
    p1Spin = new ThemeDoubleSpinBox;
    p2Spin = new ThemeDoubleSpinBox;
    for (auto* spin : {p1Spin, p2Spin}) {
        spin->setRange(-0.2, 0.2);
        spin->setDecimals(4);
        spin->setSingleStep(0.001);
    }
    p1Spin->setValue(defaults.p1);
    p2Spin->setValue(defaults.p2);
    distortionModelBox = new ThemeComboBox;
    distortionModelBox->addItems({"Radialny (k1, k2)", "Brown–Conrady (k1, k2, k3, p1, p2)"});
    distortionModelBox->setCurrentIndex(defaults.distortionModel);
    distortionModelBox->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
    distortionModelBox->setMinimumContentsLength(12);
    auto updateDistortionModel = [this](int model) {
        for (auto* spin : {k3Spin, p1Spin, p2Spin}) spin->setEnabled(model == 1);
    };
    connect(distortionModelBox, &QComboBox::currentIndexChanged, this, updateDistortionModel);
    updateDistortionModel(defaults.distortionModel);
    interpolationBox = new ThemeComboBox;
    interpolationBox->addItems({"Najbliższy sąsiad", "Dwuliniowa"});
    interpolationBox->setCurrentIndex(defaults.bilinear ? 1 : 0);
    scaleSpin = new ThemeDoubleSpinBox;
    scaleSpin->setRange(0.05, 4.0);
    scaleSpin->setDecimals(2);
    scaleSpin->setSingleStep(0.25);
    scaleSpin->setValue(defaults.scale);
    seedXSpin = new ThemeSpinBox;
    seedYSpin = new ThemeSpinBox;
    for (auto* spin : {seedXSpin, seedYSpin}) {
        spin->setRange(0, 0);
        connect(spin, &QSpinBox::valueChanged, this, &MainWindow::updateSeedMarker);
    }
    toleranceSpin = new ThemeSpinBox;
    toleranceSpin->setRange(0, 255);
    toleranceSpin->setValue(defaults.tolerance);
    localTilesCheck = new QCheckBox("GPU: kafelki w pamięci lokalnej");
    localTilesCheck->setChecked(defaults.localTiles);
    cannyLowSpin = new ThemeDoubleSpinBox;
    cannyHighSpin = new ThemeDoubleSpinBox;
    for (auto* spin : {cannyLowSpin, cannyHighSpin}) {
        spin->setRange(0.0, 1500.0);
        spin->setDecimals(1);
        spin->setSingleStep(5.0);
    }
    cannyLowSpin->setValue(defaults.cannyLow);
    cannyHighSpin->setValue(defaults.cannyHigh);
    gaborSigmaSpin = new ThemeDoubleSpinBox;
    gaborSigmaSpin->setRange(0.3, 10.0);
    gaborSigmaSpin->setDecimals(2);
    gaborSigmaSpin->setSingleStep(0.25);
    gaborSigmaSpin->setValue(defaults.gaborSigma);
    gaborWavelengthSpin = new ThemeDoubleSpinBox;
    gaborWavelengthSpin->setRange(2.0, 64.0);
    gaborWavelengthSpin->setDecimals(1);
    gaborWavelengthSpin->setSingleStep(0.5);
    gaborWavelengthSpin->setValue(defaults.gaborWavelength);
    gaborGammaSpin = new ThemeDoubleSpinBox;
    gaborGammaSpin->setRange(0.1, 2.0);
    gaborGammaSpin->setDecimals(2);
    gaborGammaSpin->setSingleStep(0.05);
    gaborGammaSpin->setValue(defaults.gaborGamma);
    gaborOrientationsSpin = new ThemeSpinBox;
    gaborOrientationsSpin->setRange(1, gaborMaxOrientations);
    gaborOrientationsSpin->setValue(defaults.gaborOrientations);
    gaborWindowLabel = note("");
    auto updateGaborWindow = [this] {
        const int radius = gaborRadius(static_cast<float>(gaborSigmaSpin->value()), static_cast<float>(gaborGammaSpin->value()));
        const int size = 2 * radius + 1;
        const int orientations = gaborOrientationsSpin->value();
        if (radius > gaborMaxRadius) {
            gaborWindowLabel->setText(QString("Okno %1 × %1 jest za duże, maksymalnie %2 × %2: zmniejsz σ albo zwiększ γ.")
                                          .arg(size).arg(2 * gaborMaxRadius + 1));
        } else {
            gaborWindowLabel->setText(QString("Okno %1 × %1: %2 orientacje × 2 maski × %3 = %4 mnożeń na piksel.")
                                          .arg(size).arg(orientations).arg(size * size).arg(2 * size * size * orientations));
        }
    };
    for (auto* spin : {gaborSigmaSpin, gaborGammaSpin}) connect(spin, &QDoubleSpinBox::valueChanged, this, updateGaborWindow);
    connect(gaborOrientationsSpin, &QSpinBox::valueChanged, this, updateGaborWindow);
    updateGaborWindow();

    parameterPages = new ParameterStack;
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
    addPage(parameterPage({{"Wariant", medianSortBox}, {"Okno", medianWindowBox},
        {"Okien na work-item", medianStripSpin}, {"", medianSwitchingCheck},
        {"", note("Sortowanie: nth_element na CPU, przez wstawianie na GPU. Sieci porównują i zamieniają piksele "
                  "zawsze w tej samej kolejności, bez rozgałęzień. Sieć współdzielona liczy pas sąsiednich okien "
                  "w pionie i raz sortuje wspólne wiersze. Mediana przełączająca zmienia tylko piksele 0 i 255 "
                  "(szum sól i pieprz), resztę zostawia.")}}));
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
    addPage(parameterPage({{"σ (szerokość obwiedni)", gaborSigmaSpin}, {"λ (długość fali, px)", gaborWavelengthSpin},
        {"γ (spłaszczenie obwiedni)", gaborGammaSpin}, {"Liczba orientacji", gaborOrientationsSpin}, {"", gaborWindowLabel},
        {"", note("Dla każdej orientacji θ = kπ/N obraz jest splatany z parą masek: kosinusową i sinusową. "
                  "Wynikiem jest największy moduł odpowiedzi √(parzysta² + nieparzysta²) po wszystkich orientacjach, "
                  "obcięty do 0…255. Promień okna r = ⌈3σ / min(γ, 1)⌉, piksele spoza obrazu powielają brzeg.")}}));
    connect(algorithmBox, &QComboBox::currentIndexChanged, this, [this, algorithmPages](int index) {
        parameterPages->setCurrentIndex(algorithmPages.at(index));
        parameterPages->updateGeometry();
    });
    connect(algorithmBox, &QComboBox::currentIndexChanged, this, &MainWindow::updateSeedMarker);

    auto* algorithmGroup = new QGroupBox("Algorytm");
    algorithmGroup->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    auto* algorithmLayout = new QVBoxLayout(algorithmGroup);
    algorithmLayout->setContentsMargins(0, 0, 0, 0);
    algorithmLayout->setSpacing(14);
    algorithmLayout->addWidget(algorithmBox);
    algorithmLayout->addWidget(parameterPages);

    cpu1Check = new QCheckBox("CPU, 1 wątek");
    cpuNCheck = new QCheckBox("CPU, wątków:");
    cpuDynamicCheck = new QCheckBox("CPU, podział dynamiczny");
    gpuBackendBox = new ThemeComboBox;
    gpuBackendBox->setObjectName("gpuBackend");
    gpuBackendBox->addItem("OpenCL");
#ifdef __APPLE__
    gpuBackendBox->addItem("Metal");
    gpuBackendBox->addItem("OpenCL i Metal");
#endif
    gpuCheck = new QCheckBox("GPU");
    gpuZeroCopyCheck = new QCheckBox("GPU (OpenCL), zero-copy");
    for (auto* check : {cpu1Check, cpuNCheck, cpuDynamicCheck, gpuCheck, gpuZeroCopyCheck}) check->setChecked(true);
    threadsSpin = new ThemeSpinBox;
    threadsSpin->setRange(2, 256);
    threadsSpin->setValue(std::max(2u, std::thread::hardware_concurrency()));
    repeatsSpin = new ThemeSpinBox;
    repeatsSpin->setRange(1, 1000);
    repeatsSpin->setValue(10);

    auto* threadsRow = new QHBoxLayout;
    threadsRow->addWidget(cpuNCheck);
    threadsRow->addWidget(threadsSpin);
    auto* modesGroup = new QGroupBox("Tryby wykonania");
    modesGroup->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Maximum);
    auto* modesLayout = new QFormLayout(modesGroup);
    modesLayout->setContentsMargins(0, 0, 0, 0);
    modesLayout->setVerticalSpacing(10);
    modesLayout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    modesLayout->addRow(cpu1Check);
    modesLayout->addRow(threadsRow);
    modesLayout->addRow(cpuDynamicCheck);
    modesLayout->addRow("Backend GPU", gpuBackendBox);
    modesLayout->addRow(gpuCheck);
    modesLayout->addRow(gpuZeroCopyCheck);
    modesLayout->addRow("Powtórzenia", repeatsSpin);
    repeatsSpin->setToolTip("Czas to mediana z powtórzeń po rozgrzewce.");
    cpuDynamicCheck->setToolTip("Te same wątki pobierają małe kawałki obrazu, więc rdzenie wydajne robią więcej.");
    gpuZeroCopyCheck->setToolTip("Wspólna pamięć CPU i GPU dla Multi-Otsu i dolin histogramu.");
    auto updateZeroCopy = [this] {
        gpuZeroCopyCheck->setEnabled(gpuBackendBox->currentIndex() != 1 && supportsZeroCopy(currentAlgorithm()));
    };
    connect(algorithmBox, &QComboBox::currentIndexChanged, this, updateZeroCopy);
    connect(gpuBackendBox, &QComboBox::currentIndexChanged, this, updateZeroCopy);
    updateZeroCopy();

    runButton = new QPushButton("Uruchom");
    runButton->setObjectName("runButton");
    runButton->setEnabled(false);
    connect(runButton, &QPushButton::clicked, this, &MainWindow::run);
    saveButton = new QPushButton("Zapisz wynik…");
    saveButton->setEnabled(false);
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::saveResult);

    threadsSpin->setMaximumWidth(86);
    auto* settings = new QWidget;
    auto* settingsLayout = new QVBoxLayout(settings);
    settingsLayout->setContentsMargins(0, 0, 8, 0);
    settingsLayout->setSpacing(24);
    settingsLayout->addWidget(algorithmGroup);
    settingsLayout->addWidget(modesGroup);
    settingsLayout->addStretch();
    auto* scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setWidget(settings);
    auto* panel = new QWidget;
    panel->setObjectName("sidebar");
    panel->setFixedWidth(336);
    auto* layout = new QVBoxLayout(panel);
    layout->setContentsMargins(20, 18, 14, 18);
    layout->setSpacing(20);
    layout->addWidget(scroll, 1);
    layout->addWidget(runButton);

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
    fileInfo->setText(QFileInfo(path).fileName());
    fileInfo->setToolTip(QFileInfo(path).absoluteFilePath());
    fileInfo->show();
    inputCaption->setText(QString("Obraz wejściowy · %1 × %2 px").arg(input.width).arg(input.height));
    seedXSpin->setRange(0, input.width - 1);
    seedYSpin->setRange(0, input.height - 1);
    seedXSpin->setValue(input.width / 2);
    seedYSpin->setValue(input.height / 2);
    updateSeedMarker();
    results.clear();
    table->setRowCount(0);
    resultPages->setCurrentIndex(0);
    resultsInfo->clear();
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
    if (gpuCheck->isChecked()) {
        const int backend = gpuBackendBox->currentIndex();
        if (backend != 1) modes.emplace_back(Mode::GPU, "GPU (OpenCL)");
        if (backend != 0) modes.emplace_back(Mode::Metal, "GPU (Metal)");
    }
    if (gpuZeroCopyCheck->isChecked() && gpuZeroCopyCheck->isEnabled()) {
        modes.emplace_back(Mode::GPUZeroCopy, "GPU (OpenCL), zero-copy");
    }
    if (modes.empty()) {
        statusBar()->showMessage("Zaznacz co najmniej jeden tryb");
        return;
    }

    runDescription = QString("%1 · mediana, n = %2").arg(algorithmNames[static_cast<int>(algorithm)]).arg(repeatsSpin->value());
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
    resultPages->setCurrentIndex(1);
    resultsInfo->setText(runDescription);
    const auto fastest = std::min_element(results.begin(), results.end(), [](const ModeResult& a, const ModeResult& b) {
        return a.timing.totalUs < b.timing.totalUs;
    });

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
            auto* item = new QTableWidgetItem(cells[column]);
            if (column >= 4) item->setToolTip(cells[column]);
            if (column >= 1 && column <= 3) item->setTextAlignment(Qt::AlignRight | Qt::AlignVCenter);
            if (column == 4 && comparison == "identyczny") item->setForeground(QColor("#356b43"));
            if (column == 1 && &result == &*fastest) {
                QFont font = item->font();
                font.setBold(true);
                item->setFont(font);
                item->setForeground(QColor("#356b43"));
                item->setToolTip("Najkrótszy czas całkowity w tym pomiarze");
            }
            table->setItem(row, column, item);
        }
    }
    const int resultsHeight = std::min(std::clamp(108 + static_cast<int>(results.size()) * 40, 190, 350),
                                       workspaceSplitter->height() / 2);
    workspaceSplitter->setSizes({workspaceSplitter->height() - resultsHeight, resultsHeight});
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
    outputCaption->setText(QString("%1 · %2 × %3 px").arg(result.name).arg(result.output.width).arg(result.output.height));
}

void MainWindow::setRunning(bool running) {
    openButton->setEnabled(!running);
    gpuBackendBox->setEnabled(!running);
    runButton->setEnabled(!running);
    runButton->setText(running ? "Obliczanie…" : "Uruchom");
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
    parameters.medianSort = medianSortBox->currentIndex();
    parameters.medianRadius = medianWindowBox->currentIndex() + 1;
    parameters.medianStrip = medianStripSpin->value();
    parameters.medianSwitching = medianSwitchingCheck->isChecked();
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
    parameters.gaborSigma = static_cast<float>(gaborSigmaSpin->value());
    parameters.gaborWavelength = static_cast<float>(gaborWavelengthSpin->value());
    parameters.gaborGamma = static_cast<float>(gaborGammaSpin->value());
    parameters.gaborOrientations = gaborOrientationsSpin->value();
    return parameters;
}
