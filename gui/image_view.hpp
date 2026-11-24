#pragma once
#include <QPixmap>
#include <QPoint>
#include <QRectF>
#include <QWidget>
#include <functional>
#include <optional>

class ImageView : public QWidget {
public:
    explicit ImageView(QWidget* parent = nullptr);
    void setImage(const QImage& image);
    void setEmptyText(const QString& text);
    void setMarker(std::optional<QPoint> point);
    void setClickHandler(std::function<void(QPoint)> handler);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    QRectF imageRect() const;

    QString emptyText = "Brak obrazu";
    QPixmap pixmap;
    QPixmap scaled;
    std::optional<QPoint> marker;
    std::function<void(QPoint)> clickHandler;
};
