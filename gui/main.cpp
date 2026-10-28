#include "main_window.hpp"
#include <QApplication>
#include <QDir>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    QDir::setCurrent(PROJECT_DIR);
    MainWindow window;
    window.resize(1400, 850);
    window.show();
    if (argc > 1) window.openImage(QString::fromLocal8Bit(argv[1]));
    return app.exec();
}
