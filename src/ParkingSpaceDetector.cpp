#include "ParkingSpaceDetector.h"

using namespace cv;

std::vector<Mat> loadImages (int sequence) {
    std::vector<String> fileNames;
    if (sequence == 0)
        glob("../data/sequence0/frames", fileNames);
    else if (sequence == 1)
        glob("./data/sequence1/frames", fileNames);
    else if (sequence == 2)
        glob("./data/sequence2/frames", fileNames);
    else if (sequence == 3)
        glob("./data/sequence3/frames", fileNames);
    else if (sequence == 4)
        glob("./data/sequence4/frames", fileNames);
    else if (sequence == 5)
        glob("./data/sequence5/frames", fileNames);

    std::vector<Mat> imgs;
    for (int i = 0; i < fileNames.size(); i++) {
        Mat img = imread(fileNames[i]);
        if (img.empty()) {
            std::cerr << "Error loading image: " << fileNames[i] << std::endl;
            continue;
        }
        imgs.push_back(img);
    }

    return imgs;
}

std::vector<Mat> preprocessImages (const std::vector<Mat>& imgs) {
    std::vector<Mat> processedImages;
    for (const auto& img: imgs) {
        Mat binaryImg;
        cvtColor(img, binaryImg, COLOR_BGR2GRAY);
        processedImages.push_back(binaryImg);
    }

    return processedImages;
}

std::vector<Mat> constrastStretch (const std::vector<Mat>& imgs) {
    std::vector<Mat> stretchedImages;
    for (const auto& img: imgs) {
        Mat stretchedImg;
        Mat img32f;
        img.convertTo(img32f, CV_32F);
        normalize(img32f, img32f, 0, 1, NORM_MINMAX);
        pow(img32f, 0.5, img32f);
        normalize(img32f, img32f, 0, 255, NORM_MINMAX);
        img32f.convertTo(stretchedImg, CV_8U);
        stretchedImages.push_back(stretchedImg);
    }

    return stretchedImages;
}

void showImages (const std::vector<Mat>& imgs) {
    for (const auto& img: imgs) {
        imshow("Image", img);
        waitKey(0);
    }
}

std::vector<Mat> detectEdges (const std::vector<Mat>& imgs, double lowThreshold, double highThreshold) {
    std::vector<Mat> edges;
    edges.reserve(imgs.size());
    for (const auto& img: imgs) {
        Mat edge;
        Canny(img, edge, lowThreshold, highThreshold);
        edges.push_back(edge);
    }

    return edges;
}

std::vector<std::vector<Vec4i>> detectLines (const std::vector<Mat>& imgs, int threshold, double minLineLength, double maxLineGap) {
    std::vector<std::vector<Vec4i>> lines;
    std::vector<Vec4i> line;
    for (const auto& img: imgs) {
        HoughLinesP(img, line, 1, CV_PI/180, threshold, minLineLength, maxLineGap);
        // HoughLines(img, line, 1, CV_PI/180, threshold);
        lines.push_back(line);
    }

    return lines;
}

// Filter the lines 
std::vector<std::vector<Vec4i>> filterLines (const std::vector<std::vector<Vec4i>>& linesVector, const double minAngle, const double maxAngle, const double minLength, const double maxLength, const double minDistance, const double maxDistance) {
    std::vector<std::vector<Vec4i>> filteredLinesVector;

   for (const auto& lines: linesVector) {
        std::vector<Vec4i> filteredLines;
        for (size_t i = 0; i < lines.size(); i++) {
            Vec4i l = lines[i];
            double dx = l[2] - l[0];
            double dy = l[3] - l[1];
            double angle = atan2(dy, dx) * 180 / CV_PI;
            double length = sqrt(dx * dx + dy * dy);

            if (abs(abs(angle) - 90) >= minAngle && abs(angle) - 90 <= maxAngle){//&& length >= minLength && length <= maxLength) {
                filteredLines.push_back(l);
            }
        }

        // std::vector<Vec4i> finalLines;
        // for (size_t i = 0; i < filteredLines.size(); i++) {
        //     Vec4i l1 = filteredLines[i];
        //     bool keepLine = true;

        //     for (size_t j = i + 1; j < filteredLines.size(); j++) {
        //         Vec4i l2 = filteredLines[j];
        //         double distance = norm(Point(l1[0], l1[1]) - Point(l2[0], l2[1]));

        //         if (distance >= minDistance && distance <= maxDistance) {
        //             finalLines.push_back(l1);
        //             finalLines.push_back(l2);
        //             keepLine = false;
        //             break;
        //         }
        //     }

        //     if (keepLine) {
        //         finalLines.push_back(l1);
        //     }
        // }

        filteredLinesVector.push_back(filteredLines);
   }
    return filteredLinesVector;
}



void drawLines (std::vector<Mat>& imgs, const std::vector<std::vector<Vec4i>>& lines) {
    for (const auto& img: imgs) {
        for (const auto& line: lines) {
            for (const auto& l: line) {
                cv::line(img, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
            }
        }
    }
}

void drawBoundingBoxes (std::vector<Mat>& imgs, const std::vector<std::vector<Vec4i>>& linesVector) {
    for (const auto& img: imgs) {
        for (const auto& lines: linesVector) {
            for (size_t i = 0; i < lines.size(); i++) {
                for (size_t j = 0; j < lines.size(); i++) {
                    Vec4i l1 = lines[i];
                    Vec4i l2 = lines[j];

                    double dx1 = l1[2] - l1[0];
                    double dy1 = l1[3] - l1[1];
                    double angle1 = atan2(dy1, dx1) * 180 / CV_PI;

                    double dx2 = l2[2] - l2[0];
                    double dy2 = l2[3] - l2[1];
                    double angle2 = atan2(dy2, dx2) * 180 / CV_PI;

                    if (abs(angle1 - angle2) < 10.0) {
                        Point2f p1 (l1[0], l1[1]);
                        Point2f p2 (l1[2], l1[3]);
                        Point2f p3 (l2[0], l2[1]);
                        Point2f p4 (l2[2], l2[3]);

                        std::vector<Point2f> points = {p1, p2, p3, p4};
                        sort(points.begin(), points.end(), [](const Point2f& p1, const Point2f& p2) {
                            return p1.x < p2.x;
                        });

                        Point2f topLeft = points[0];
                        Point2f topRight = points[1];
                        Point2f bottomLeft = points[2];
                        Point2f bottomRight = points[3];

                        Point2f center ((topLeft.x + topRight.x) / 2, (topLeft.y + bottomLeft.y) / 2);
                        Size2f size (abs(topRight.x - topLeft.x), abs(bottomLeft.y - topLeft.y));
                        double angle = atan2(dy1, dx1) * 180 / CV_PI;

                        RotatedRect rect (center, size, angle);

                        Point2f vertices[4];
                        rect.points(vertices);
                        for (int i = 0; i < 4; i++) {
                            line(img, vertices[i], vertices[(i + 1) % 4], Scalar(0, 0, 255), 3, LINE_AA);
                        }
                    }
                }
            }
        }
    }
}