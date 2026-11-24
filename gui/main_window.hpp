#pragma once
#include "algorithms.hpp"
#include "image.hpp"
#include <QFutureWatcher>
#include <QMainWindow>
#include <QString>
#include <vector>

class ImageView;
class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QLabel;
class QPushButton;
class QSpinBox;
class QSplitter;
class QStackedWidget;
class QTableWidget;

struct ModeResult {
    Mode mode;
    QString name;
    Timing timing;
    Image output;
};

struct RunOutcome {
    std::vector<ModeResult> results;
    QString error;
};

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow();
    void openImage(const QString& path);

private:
    QWidget* createControls();
    void chooseImage();
    void saveResult();
    void run();
    void runFinished();
    void showResult(int row);
    void setRunning(bool running);
    void updateSeedMarker();
    Algorithm currentAlgorithm() const;
    Parameters currentParameters() const;

    Image input;
    std::vector<ModeResult> results;
    QFutureWatcher<RunOutcome> watcher;

    ImageView* inputView;
    ImageView* outputView;
    QLabel* inputCaption;
    QLabel* outputCaption;
    QLabel* fileInfo;
    QLabel* resultsInfo;
    QStackedWidget* resultPages;
    QSplitter* workspaceSplitter;
    QString runDescription;
    QTableWidget* table;

    QPushButton* openButton;
    QPushButton* runButton;
    QPushButton* saveButton;
    QComboBox* algorithmBox;
    QStackedWidget* parameterPages;
    QSpinBox* thresholdSpin;
    QDoubleSpinBox* contrastFactorSpin;
    QSpinBox* classesSpin;
    QSpinBox* maxClassesSpin;
    QSpinBox* radiusSpin;
    QSpinBox* contrastSpin;
    QSpinBox* localRadiusSpin;
    QDoubleSpinBox* localThresholdKSpin;
    QComboBox* filterBox;
    QComboBox* medianSortBox;
    QComboBox* medianWindowBox;
    QSpinBox* medianStripSpin;
    QCheckBox* medianSwitchingCheck;
    QComboBox* distortionModelBox;
    QDoubleSpinBox* k1Spin;
    QDoubleSpinBox* k2Spin;
    QDoubleSpinBox* k3Spin;
    QDoubleSpinBox* p1Spin;
    QDoubleSpinBox* p2Spin;
    QComboBox* interpolationBox;
    QDoubleSpinBox* scaleSpin;
    QSpinBox* seedXSpin;
    QSpinBox* seedYSpin;
    QSpinBox* toleranceSpin;
    QCheckBox* localTilesCheck;
    QDoubleSpinBox* cannyLowSpin;
    QDoubleSpinBox* cannyHighSpin;
    QDoubleSpinBox* gaborSigmaSpin;
    QDoubleSpinBox* gaborWavelengthSpin;
    QDoubleSpinBox* gaborGammaSpin;
    QSpinBox* gaborOrientationsSpin;
    QLabel* gaborWindowLabel;
    QCheckBox* cpu1Check;
    QCheckBox* cpuNCheck;
    QCheckBox* cpuDynamicCheck;
    QComboBox* gpuBackendBox;
    QCheckBox* gpuCheck;
    QCheckBox* gpuZeroCopyCheck;
    QSpinBox* threadsSpin;
    QSpinBox* repeatsSpin;
};
