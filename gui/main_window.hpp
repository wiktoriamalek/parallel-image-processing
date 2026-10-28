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
    Algorithm currentAlgorithm() const;
    Parameters currentParameters() const;

    Image input;
    std::vector<ModeResult> results;
    QFutureWatcher<RunOutcome> watcher;

    ImageView* inputView;
    ImageView* outputView;
    QLabel* inputCaption;
    QLabel* outputCaption;
    QTableWidget* table;

    QPushButton* openButton;
    QPushButton* runButton;
    QPushButton* saveButton;
    QComboBox* algorithmBox;
    QStackedWidget* parameterPages;
    QSpinBox* thresholdSpin;
    QSpinBox* radiusSpin;
    QSpinBox* contrastSpin;
    QComboBox* filterBox;
    QCheckBox* sortingNetworkCheck;
    QDoubleSpinBox* k1Spin;
    QDoubleSpinBox* k2Spin;
    QCheckBox* cpu1Check;
    QCheckBox* cpuNCheck;
    QCheckBox* gpuCheck;
    QSpinBox* threadsSpin;
    QSpinBox* repeatsSpin;
};
