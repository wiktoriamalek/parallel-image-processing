#include "image_view.hpp"
#include <QMouseEvent>
#include <QPainter>
#include <algorithm>
#include <utility>

ImageView::ImageView(QWidget* parent) : QWidget(parent) {
    setMinimumSize(200, 200);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

void ImageView::setImage(const QImage& image) {
    pixmap = QPixmap::fromImage(image);
    scaled = QPixmap();
    update();
}

void ImageView::setMarker(std::optional<QPoint> point) {
    marker = point;
    update();
}

void ImageView::setClickHandler(std::function<void(QPoint)> handler) {
    clickHandler = std::move(handler);
}

QRectF ImageView::imageRect() const {
    QSizeF shown = QSizeF(pixmap.size().scaled(size() * devicePixelRatioF(), Qt::KeepAspectRatio)) / devicePixelRatioF();
    return QRectF(QPointF((width() - shown.width()) / 2, (height() - shown.height()) / 2), shown);
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
    if (marker) {
        QRectF area = imageRect();
        QPointF center(area.left() + (marker->x() + 0.5) * area.width() / pixmap.width(),
                       area.top() + (marker->y() + 0.5) * area.height() / pixmap.height());
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setPen(QPen(Qt::red, 2));
        painter.drawEllipse(center, 6, 6);
        painter.drawLine(center - QPointF(10, 0), center + QPointF(10, 0));
        painter.drawLine(center - QPointF(0, 10), center + QPointF(0, 10));
    }
}

void ImageView::mousePressEvent(QMouseEvent* event) {
    if (pixmap.isNull() || !clickHandler) return;
    QRectF area = imageRect();
    QPointF position = event->position();
    if (!area.contains(position)) return;
    int x = static_cast<int>((position.x() - area.left()) * pixmap.width() / area.width());
    int y = static_cast<int>((position.y() - area.top()) * pixmap.height() / area.height());
    clickHandler(QPoint(std::clamp(x, 0, pixmap.width() - 1), std::clamp(y, 0, pixmap.height() - 1)));
}
