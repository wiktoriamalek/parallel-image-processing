#pragma once
#include <QPixmap>
#include <QWidget>

class ImageView : public QWidget {
public:
    explicit ImageView(QWidget* parent = nullptr);
    void setImage(const QImage& image);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    QPixmap pixmap;
    QPixmap scaled;
};
