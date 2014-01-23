#include "qt_opencv.hpp"

QImage mat2QImage(cv::Mat &mat){
    IplImage input(mat);
    QImage image(input.width, input.height, QImage::Format_RGB32);
    uchar* pBits = image.bits();
    int nBytesPerLine = image.bytesPerLine();
    for (int n = 0; n < input.height; n++){
        for (int m = 0; m < input.width; m++){
            CvScalar s = cvGet2D(&input, n, m);
            QRgb value = qRgb((uchar)s.val[2], (uchar)s.val[1], (uchar)s.val[0]);

            uchar* scanLine = pBits + n * nBytesPerLine;
            ((uint*)scanLine)[m] = value;
        }
    }
    return image;
}
