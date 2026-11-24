#pragma once

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QSpinBox>

class ThemeComboBox : public QComboBox {
public:
    using QComboBox::QComboBox;

protected:
    void paintEvent(QPaintEvent* event) override;
};

class ThemeSpinBox : public QSpinBox {
public:
    using QSpinBox::QSpinBox;

protected:
    void paintEvent(QPaintEvent* event) override;
};

class ThemeDoubleSpinBox : public QDoubleSpinBox {
public:
    using QDoubleSpinBox::QDoubleSpinBox;

protected:
    void paintEvent(QPaintEvent* event) override;
};

void applyTheme(QWidget& window);
