#include "theme.hpp"
#include <QApplication>
#include <QFontDatabase>
#include <QPalette>
#include <QPainter>
#include <QPainterPath>
#include <QProxyStyle>
#include <QStyleOption>
#include <QStyleFactory>
#include <QWidget>

namespace {
void paintArrow(QPainter& painter, QPointF center, qreal direction, bool enabled) {
    painter.setRenderHint(QPainter::Antialiasing);
    const QColor color = enabled ? QColor("#515b51") : QColor("#a3aaa0");
    painter.setPen(QPen(color, 1.5, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    QPainterPath path;
    path.moveTo(center + QPointF(-3, -1.5 * direction));
    path.lineTo(center + QPointF(0, 1.5 * direction));
    path.lineTo(center + QPointF(3, -1.5 * direction));
    painter.drawPath(path);
}

class AppStyle : public QProxyStyle {
public:
    AppStyle() : QProxyStyle(QStyleFactory::create("Fusion")) {}

    void drawPrimitive(PrimitiveElement element, const QStyleOption* option, QPainter* painter,
                       const QWidget* widget = nullptr) const override {
        if (element == PE_IndicatorCheckBox) {
            painter->save();
            painter->setRenderHint(QPainter::Antialiasing);
            const QRectF rect = QRectF(option->rect).adjusted(1, 1, -1, -1);
            const bool checked = option->state & (State_On | State_NoChange);
            const bool enabled = option->state & State_Enabled;
            const QColor fill = checked ? (enabled ? QColor("#42664d") : QColor("#cbd5c8")) : QColor("#ffffff");
            const QColor border = checked ? fill : QColor("#929d8f");
            painter->setPen(QPen(border, 1));
            painter->setBrush(fill);
            painter->drawRoundedRect(rect, 3, 3);
            if (checked) {
                painter->setPen(QPen(enabled ? QColor("#ffffff") : QColor("#7d8d78"), 1.8,
                                    Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
                QPainterPath path;
                if (option->state & State_NoChange) {
                    path.moveTo(rect.left() + 4, rect.center().y());
                    path.lineTo(rect.right() - 4, rect.center().y());
                } else {
                    path.moveTo(rect.left() + 3, rect.center().y());
                    path.lineTo(rect.left() + 6, rect.bottom() - 4);
                    path.lineTo(rect.right() - 3, rect.top() + 4);
                }
                painter->drawPath(path);
            }
            painter->restore();
            return;
        }
        QProxyStyle::drawPrimitive(element, option, painter, widget);
    }
};
}

void ThemeComboBox::paintEvent(QPaintEvent* event) {
    QComboBox::paintEvent(event);
    QPainter painter(this);
    paintArrow(painter, QPointF(width() - 14, height() / 2.0), 1, isEnabled());
}

void ThemeSpinBox::paintEvent(QPaintEvent* event) {
    QSpinBox::paintEvent(event);
    QPainter painter(this);
    paintArrow(painter, QPointF(width() - 12, height() * 0.29), -1, isEnabled() && value() < maximum());
    paintArrow(painter, QPointF(width() - 12, height() * 0.71), 1, isEnabled() && value() > minimum());
}

void ThemeDoubleSpinBox::paintEvent(QPaintEvent* event) {
    QDoubleSpinBox::paintEvent(event);
    QPainter painter(this);
    paintArrow(painter, QPointF(width() - 12, height() * 0.29), -1, isEnabled() && value() < maximum());
    paintArrow(painter, QPointF(width() - 12, height() * 0.71), 1, isEnabled() && value() > minimum());
}

void applyTheme(QWidget& window) {
    QApplication::setStyle(new AppStyle);
    QFont font = QFontDatabase::systemFont(QFontDatabase::GeneralFont);
    font.setPixelSize(13);
    window.setFont(font);

    QPalette palette;
    palette.setColor(QPalette::Window, QColor("#f1f2ed"));
    palette.setColor(QPalette::WindowText, QColor("#283128"));
    palette.setColor(QPalette::Base, QColor("#ffffff"));
    palette.setColor(QPalette::AlternateBase, QColor("#f6f7f3"));
    palette.setColor(QPalette::Text, QColor("#283128"));
    palette.setColor(QPalette::Button, QColor("#ffffff"));
    palette.setColor(QPalette::ButtonText, QColor("#283128"));
    palette.setColor(QPalette::Highlight, QColor("#42664d"));
    palette.setColor(QPalette::HighlightedText, QColor("#ffffff"));
    palette.setColor(QPalette::ToolTipBase, QColor("#ffffff"));
    palette.setColor(QPalette::ToolTipText, QColor("#283128"));
    palette.setColor(QPalette::Disabled, QPalette::Text, QColor("#8c9588"));
    palette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor("#8c9588"));
    palette.setColor(QPalette::Disabled, QPalette::WindowText, QColor("#8c9588"));
    palette.setColor(QPalette::PlaceholderText, QColor("#626e5d"));
    QApplication::setPalette(palette);
    window.setPalette(palette);

    window.setStyleSheet(R"(
        QMainWindow, QWidget#appRoot { background: #f1f2ed; color: #283128; }
        QWidget#topBar { background: #f1f2ed; }
        QWidget#sidebar, QFrame#previewCard, QFrame#resultsCard {
            background: #ffffff;
            border: 1px solid #d7dcd1;
            border-radius: 12px;
        }
        QLabel { color: #424c40; background: transparent; border: none; }
        QLabel#appTitle { color: #252f24; font-size: 24px; font-weight: 600; }
        QLabel#fileInfo { color: #626e5d; font-size: 12px; }
        QLabel#panelHeading { color: #303a2d; font-size: 14px; font-weight: 600; }
        QLabel#hint, QLabel#emptyResults { color: #626e5d; font-size: 12px; }
        QLabel#hint { padding-top: 4px; }
        QGroupBox {
            color: #303a2d;
            font-weight: 600;
            border: none;
            padding-top: 24px;
            margin-top: 2px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 0px; top: 1px; }
        QComboBox, QSpinBox, QDoubleSpinBox {
            color: #303a2d;
            background: #ffffff;
            border: 1px solid #c6cebf;
            border-radius: 7px;
            padding: 7px 10px;
            min-height: 20px;
            selection-background-color: #42664d;
            selection-color: #ffffff;
        }
        QComboBox { padding-right: 28px; }
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #8e9f86; }
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #42664d; }
        QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
            color: #8c9588; background: #f0f2ec; border-color: #dce1d6;
        }
        QComboBox::drop-down { border: none; width: 26px; }
        QComboBox::down-arrow { image: none; }
        QComboBox QAbstractItemView {
            color: #303a2d; background: #ffffff;
            border: 1px solid #b8c4b0;
            selection-background-color: #dfe9db;
            selection-color: #283128;
            outline: none;
        }
        QComboBox QAbstractItemView::item { padding: 7px; min-height: 22px; }
        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-origin: border; subcontrol-position: top right;
            width: 23px; border: none; background: transparent;
        }
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border; subcontrol-position: bottom right;
            width: 23px; border: none; background: transparent;
        }
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: none; }
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow { image: none; }
        QCheckBox { color: #424c40; spacing: 9px; padding: 4px 0px; font-weight: 400; }
        QCheckBox:disabled { color: #8c9588; }
        QCheckBox::indicator { width: 17px; height: 17px; }
        QPushButton {
            color: #303a2d; background: #f8f9f5;
            border: 1px solid #c6cebf;
            border-radius: 8px;
            padding: 9px 17px;
            min-height: 20px;
            font-weight: 500;
        }
        QPushButton:hover { background: #edf1e8; border-color: #9bab91; }
        QPushButton:pressed { background: #e0e7d9; }
        QPushButton:disabled { color: #8c9588; background: #f6f7f3; border-color: #dce1d6; }
        QPushButton#runButton {
            color: #ffffff; background: #42664d; border-color: #42664d;
            min-height: 27px; font-size: 14px; font-weight: 600;
        }
        QPushButton#runButton:hover { background: #507858; border-color: #507858; }
        QPushButton#runButton:pressed { background: #35543e; }
        QPushButton#runButton:disabled { color: #82917c; background: #dce5d6; border-color: #cdd9c6; }
        QScrollArea { background: transparent; border: none; }
        QScrollArea > QWidget > QWidget { background: transparent; }
        QScrollBar:vertical { background: transparent; width: 7px; margin: 2px 0px; }
        QScrollBar::handle:vertical { background: #bbc6b3; border-radius: 3px; min-height: 30px; }
        QScrollBar::handle:vertical:hover { background: #94a388; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        QScrollBar:horizontal { background: transparent; height: 7px; }
        QScrollBar::handle:horizontal { background: #bbc6b3; border-radius: 3px; min-width: 30px; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
        QTableWidget {
            color: #394534; background: #ffffff; alternate-background-color: #f6f7f3;
            border: none; border-radius: 0px; selection-background-color: #dfe9db;
            selection-color: #283128; outline: none;
        }
        QTableWidget::item { padding: 10px 12px; border-bottom: 1px solid #e3e7dd; }
        QTableWidget::item:selected { background: #dfe9db; color: #283128; }
        QHeaderView::section {
            color: #596653; background: #f0f3eb;
            padding: 11px 12px; border: none;
            border-bottom: 1px solid #d4ddcc; font-size: 12px; font-weight: 500;
        }
        QSplitter::handle { background: transparent; }
        QSplitter::handle:hover { background: #d6dfcd; border-radius: 3px; }
        QStatusBar { color: #626e5d; background: #f1f2ed; font-size: 12px; padding: 4px 16px; }
        QStatusBar::item { border: none; }
        QToolTip { color: #283128; background: #ffffff; border: 1px solid #b8c4b0; padding: 7px; }
    )");
}
