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
    const QColor color = enabled ? QColor("#b6c8e0") : QColor("#596c87");
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
            const QColor fill = checked ? (enabled ? QColor("#62d5bc") : QColor("#344d49")) : QColor("#1b283b");
            const QColor border = checked ? fill : QColor("#4b5f7b");
            painter->setPen(QPen(border, 1));
            painter->setBrush(fill);
            painter->drawRoundedRect(rect, 3, 3);
            if (checked) {
                painter->setPen(QPen(enabled ? QColor("#102e25") : QColor("#82a39d"), 1.8,
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
    palette.setColor(QPalette::Window, QColor("#0c121d"));
    palette.setColor(QPalette::WindowText, QColor("#e8eef7"));
    palette.setColor(QPalette::Base, QColor("#192436"));
    palette.setColor(QPalette::AlternateBase, QColor("#172233"));
    palette.setColor(QPalette::Text, QColor("#e8eef7"));
    palette.setColor(QPalette::Button, QColor("#1b293c"));
    palette.setColor(QPalette::ButtonText, QColor("#e8eef7"));
    palette.setColor(QPalette::Highlight, QColor("#62d5bc"));
    palette.setColor(QPalette::HighlightedText, QColor("#10251f"));
    palette.setColor(QPalette::ToolTipBase, QColor("#26374d"));
    palette.setColor(QPalette::ToolTipText, QColor("#e8eef7"));
    palette.setColor(QPalette::Disabled, QPalette::Text, QColor("#61718a"));
    palette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor("#61718a"));
    window.setPalette(palette);

    window.setStyleSheet(R"(
        QMainWindow, QWidget#appRoot { background: #0c121d; color: #e8eef7; }
        QWidget#topBar { background: #0c121d; }
        QWidget#sidebar, QFrame#previewCard, QFrame#resultsCard {
            background: #131d2c;
            border: 1px solid #253248;
            border-radius: 12px;
        }
        QLabel { color: #c4d1e3; background: transparent; border: none; }
        QLabel#appTitle { color: #f1f5fb; font-size: 24px; font-weight: 600; }
        QLabel#fileInfo { color: #8fA2bd; font-size: 12px; }
        QLabel#panelHeading { color: #e4edf9; font-size: 14px; font-weight: 600; }
        QLabel#hint, QLabel#emptyResults { color: #8fa2bc; font-size: 12px; }
        QLabel#hint { padding-top: 4px; }
        QGroupBox {
            color: #e1e9f5;
            font-weight: 600;
            border: none;
            padding-top: 24px;
            margin-top: 2px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 0px; top: 1px; }
        QComboBox, QSpinBox, QDoubleSpinBox {
            color: #e4edf9;
            background: #1b283b;
            border: 1px solid #33435b;
            border-radius: 7px;
            padding: 7px 10px;
            min-height: 20px;
            selection-background-color: #356358;
        }
        QComboBox { padding-right: 28px; }
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #4b617e; }
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #62d5bc; }
        QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
            color: #62738d; background: #172131; border-color: #28364b;
        }
        QComboBox::drop-down { border: none; width: 26px; }
        QComboBox::down-arrow { image: none; }
        QComboBox QAbstractItemView {
            color: #e4edf9; background: #1b283b;
            border: 1px solid #435775;
            selection-background-color: #31584f;
            selection-color: #effffb;
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
        QCheckBox { color: #c4d1e3; spacing: 9px; padding: 4px 0px; font-weight: 400; }
        QCheckBox:disabled { color: #62738d; }
        QCheckBox::indicator { width: 17px; height: 17px; }
        QPushButton {
            color: #e2ebf7; background: #1c2a3e;
            border: 1px solid #354861;
            border-radius: 8px;
            padding: 9px 17px;
            min-height: 20px;
            font-weight: 500;
        }
        QPushButton:hover { background: #283b53; border-color: #526b8a; }
        QPushButton:pressed { background: #162436; }
        QPushButton:disabled { color: #5f728e; background: #172233; border-color: #27364c; }
        QPushButton#runButton {
            color: #0c2922; background: #62d5bc; border-color: #62d5bc;
            min-height: 27px; font-size: 14px; font-weight: 600;
        }
        QPushButton#runButton:hover { background: #83e2cd; border-color: #83e2cd; }
        QPushButton#runButton:pressed { background: #42bfa4; }
        QPushButton#runButton:disabled { color: #6f9e96; background: #233e3c; border-color: #2c514c; }
        QScrollArea { background: transparent; border: none; }
        QScrollArea > QWidget > QWidget { background: transparent; }
        QScrollBar:vertical { background: transparent; width: 7px; margin: 2px 0px; }
        QScrollBar::handle:vertical { background: #3b4d68; border-radius: 3px; min-height: 30px; }
        QScrollBar::handle:vertical:hover { background: #597190; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        QScrollBar:horizontal { background: transparent; height: 7px; }
        QScrollBar::handle:horizontal { background: #3b4d68; border-radius: 3px; min-width: 30px; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
        QTableWidget {
            color: #c6d4e8; background: #131d2c; alternate-background-color: #172233;
            border: none; border-radius: 0px; selection-background-color: #254438;
            selection-color: #f2fffb; outline: none;
        }
        QTableWidget::item { padding: 10px 12px; border-bottom: 1px solid #213047; }
        QTableWidget::item:selected { background: #23483f; color: #edfff9; }
        QHeaderView::section {
            color: #98abc5; background: #182538;
            padding: 11px 12px; border: none;
            border-bottom: 1px solid #2c3e56; font-size: 12px; font-weight: 500;
        }
        QSplitter::handle { background: transparent; }
        QSplitter::handle:hover { background: #2a3c53; border-radius: 3px; }
        QStatusBar { color: #8fa2bd; background: #0c121d; font-size: 12px; padding: 4px 16px; }
        QStatusBar::item { border: none; }
        QToolTip { color: #e8eef7; background: #26374d; border: 1px solid #4d6380; padding: 7px; }
    )");
}
