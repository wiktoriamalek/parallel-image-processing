#include "image_view.hpp"
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
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

void ImageView::setEmptyText(const QString& text) {
    emptyText = text;
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
    const QRectF content = QRectF(rect()).adjusted(14, 14, -14, -14);
    const QSize available = (content.size() * devicePixelRatioF()).toSize();
    const QSizeF shown = QSizeF(pixmap.size().scaled(available, Qt::KeepAspectRatio)) / devicePixelRatioF();
    return QRectF(QPointF(content.center().x() - shown.width() / 2,
                         content.center().y() - shown.height() / 2), shown);
}

void ImageView::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    QPainterPath clip;
    clip.addRoundedRect(QRectF(rect()), 8, 8);
    painter.setClipPath(clip);
    painter.fillRect(rect(), QColor("#0b111b"));
    if (pixmap.isNull()) {
        const QPointF center = QRectF(rect()).center() - QPointF(0, 20);
        const QRectF icon(center.x() - 24, center.y() - 20, 48, 40);
        painter.setPen(QPen(QColor("#3d526f"), 1.5));
        painter.drawRoundedRect(icon, 5, 5);
        painter.drawEllipse(QPointF(icon.left() + 13, icon.top() + 12), 3, 3);
        QPainterPath mountains;
        mountains.moveTo(icon.left() + 6, icon.bottom() - 8);
        mountains.lineTo(icon.left() + 19, icon.top() + 20);
        mountains.lineTo(icon.left() + 27, icon.top() + 28);
        mountains.lineTo(icon.left() + 34, icon.top() + 20);
        mountains.lineTo(icon.right() - 5, icon.bottom() - 8);
        painter.drawPath(mountains);
        painter.setPen(QColor("#8fa2bd"));
        painter.drawText(QRectF(16, center.y() + 34, width() - 32, 50), Qt::AlignHCenter | Qt::AlignTop | Qt::TextWordWrap, emptyText);
        return;
    }
    const qreal ratio = devicePixelRatioF();
    const QRectF area = imageRect();
    const QSize target = (area.size() * ratio).toSize();
    if (scaled.size() != target) {
        const Qt::TransformationMode mode = target.width() > pixmap.width() ? Qt::FastTransformation : Qt::SmoothTransformation;
        scaled = pixmap.scaled(target, Qt::KeepAspectRatio, mode);
        scaled.setDevicePixelRatio(ratio);
    }
    painter.setRenderHint(QPainter::Antialiasing, false);
    painter.drawPixmap(area.topLeft(), scaled);
    if (marker) {
        const QPointF center(area.left() + (marker->x() + 0.5) * area.width() / pixmap.width(),
                             area.top() + (marker->y() + 0.5) * area.height() / pixmap.height());
        painter.setRenderHint(QPainter::Antialiasing);
        for (const QPen& pen : {QPen(QColor("#0b111b"), 4), QPen(QColor("#ffbd79"), 2)}) {
            painter.setPen(pen);
            painter.drawEllipse(center, 6, 6);
            painter.drawLine(center - QPointF(11, 0), center + QPointF(11, 0));
            painter.drawLine(center - QPointF(0, 11), center + QPointF(0, 11));
        }
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
