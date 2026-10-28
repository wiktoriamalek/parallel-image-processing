#include "image_view.hpp"
#include <QPainter>

ImageView::ImageView(QWidget* parent) : QWidget(parent) {
    setMinimumSize(200, 200);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void ImageView::setImage(const QImage& image) {
    pixmap = QPixmap::fromImage(image);
    scaled = QPixmap();
    update();
}

void ImageView::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    if (pixmap.isNull()) {
        painter.drawText(rect(), Qt::AlignCenter, "Brak obrazu");
        return;
    }
    qreal ratio = devicePixelRatioF();
    QSize target = pixmap.size().scaled(size() * ratio, Qt::KeepAspectRatio);
    if (scaled.size() != target) {
        Qt::TransformationMode mode = target.width() > pixmap.width() ? Qt::FastTransformation : Qt::SmoothTransformation;
        scaled = pixmap.scaled(target, Qt::KeepAspectRatio, mode);
        scaled.setDevicePixelRatio(ratio);
    }
    QSizeF shown = QSizeF(scaled.size()) / ratio;
    painter.drawPixmap(QPointF((width() - shown.width()) / 2, (height() - shown.height()) / 2), scaled);
}
