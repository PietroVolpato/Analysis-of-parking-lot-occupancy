#include "ParkingSpaceDetector.h"

using namespace cv;

std::vector<Mat> loadImages (const int sequence) {
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

std::vector<LineParams> computeLineParams(const std::vector<Vec4i>& lines) {
    std::vector<LineParams> params;
    for (const auto& line : lines) {
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        double a = y1 - y2;
        double b = x2 - x1;
        double c = x1 * y2 - x2 * y1;
        double theta = atan2(b, a);
        double rho = -c / sqrt(a*a + b*b);
        double length = sqrt((x2 - x1)*(x2 - x1) + (y2 - y1)*(y2 - y1));
        params.push_back({rho, theta, line, length});
    }
    return params;
}

Mat preprocessImage (const Mat& img) {
    // Mat gray, blurred, edges;
    // cvtColor(img, gray, COLOR_BGR2GRAY);
    // // adaptiveThreshold(gray, gray, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    // normalize(gray, gray, 0, 255, NORM_MINMAX);
    // bilateralFilter(gray, blurred, 9, 75, 75);
    // // GaussianBlur(gray, blurred, Size(5, 5), 0);
    // Canny(blurred, edges, 50, 150);

    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat blurred;
    GaussianBlur(gray, blurred, Size(3, 3), 1);
    // bilateralFilter(gray, blurred, 5, 75, 75);
    Mat thresh;
    adaptiveThreshold(blurred, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 7, 9);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernel);

    Mat edges;
    Canny(thresh, edges, 50, 150);
    // return blurred;
    // Mat gray;
    // cvtColor(img, gray, COLOR_BGR2GRAY);
    // Mat equalized;
    // equalizeHist(gray, equalized);
    // Mat thresh;
    // threshold(equalized, thresh, 245, 255, THRESH_BINARY);
    // adaptiveThreshold(equalized, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, 0);
    // // GaussianBlur(thresh, thresh, Size(5, 5), 0);
    // Mat blurred;
    // bilateralFilter(thresh, blurred, 9, 75, 75);
    // Mat edges;
    // Canny(blurred, edges, 500, 1500);

    return thresh;
}

void showImage (const Mat& imgs) {
    imshow("Image", imgs);
    waitKey(0);
}

std::vector<Vec4i> detectLines (const Mat& edges, int threshold, double minLineLength, double maxLineGap) {
    std::vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI/180, threshold, minLineLength, maxLineGap);
    return lines;
}

std::vector<std::vector<Point>> findContours (const Mat& edges, const Mat& img) {
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++) {
        Scalar color = Scalar(0, 0, 255);
        drawContours(img, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
    }

    imshow("Contours", img);
    waitKey(0);

    return contours;
}

void drawLines (Mat& imgs, const std::vector<LineParams>& lines) {
    for (const auto& line : lines) {
        Point p1(line.endpoints[0], line.endpoints[1]);
        Point p2(line.endpoints[2], line.endpoints[3]);
        cv::line(imgs, p1, p2, Scalar(0, 0, 255), 2);
    }
}

std::vector<LineParams> filterLines (std::vector<LineParams>& lines) {
    std::vector<LineParams> filteredLines;
    for (const auto& line : lines) {
        if ((abs(line.theta) > 4*CV_PI/9 && abs(line.theta) < 5*CV_PI/9 ) || (abs(line.theta - CV_PI/2) < CV_PI/9 )) {
            filteredLines.push_back(line);
        }
    }
    return filteredLines;
}

std::vector<std::vector<LineParams>> clusterLinesByTheta(const std::vector<LineParams>& lines, double thetaThreshold) {
    std::vector<std::vector<LineParams>> clusters;
    if (lines.empty()) return clusters;

    std::vector<LineParams> sortedLines = lines;
    sort(sortedLines.begin(), sortedLines.end(), [](const LineParams& a, const LineParams& b) {
        return a.theta < b.theta;
    });

    std::vector<LineParams> currentCluster = {sortedLines[0]};
    for (size_t i = 1; i < sortedLines.size(); ++i) {
        double deltaTheta = abs(sortedLines[i].theta - currentCluster.back().theta);
        deltaTheta = min(deltaTheta, CV_2PI - deltaTheta);
        if (deltaTheta <= thetaThreshold) {
            currentCluster.push_back(sortedLines[i]);
        } else {
            clusters.push_back(currentCluster);
            currentCluster = {sortedLines[i]};
        }
    }
    if (!currentCluster.empty()) {
        clusters.push_back(currentCluster);
    }
    return clusters;
}

std::vector<RotatedRect> detectParkingSpaces(const std::vector<LineParams>& lineParams) {
    double thetaThreshold = 5 * CV_PI / 180;
    std::vector<std::vector<LineParams>> clusters = clusterLinesByTheta(lineParams, thetaThreshold);

    std::vector<RotatedRect> parkingSpaces;

    for (const auto& cluster : clusters) {
        if (cluster.size() < 2) continue;

        std::vector<LineParams> sortedCluster = cluster;
        sort(sortedCluster.begin(), sortedCluster.end(), [](const LineParams& a, const LineParams& b) {
            return a.rho < b.rho;
        });

        std::vector<double> diffs;
        for (size_t i = 1; i < sortedCluster.size(); ++i) {
            diffs.push_back(sortedCluster[i].rho - sortedCluster[i-1].rho);
        }
        if (diffs.empty()) continue;

        sort(diffs.begin(), diffs.end());
        double medianDiff = diffs[diffs.size() / 2];
        double tolerance = 0.2 * medianDiff;

        for (size_t i = 0; i < sortedCluster.size() - 1; ++i) {
            double diff = sortedCluster[i+1].rho - sortedCluster[i].rho;
            if (abs(diff - medianDiff) <= tolerance) {
                const LineParams& line1 = sortedCluster[i];
                const LineParams& line2 = sortedCluster[i+1];

                Vec4i l1 = line1.endpoints;
                Vec4i l2 = line2.endpoints;
                Point2f mid1((l1[0]+l1[2])/2.0, (l1[1]+l1[3])/2.0);
                Point2f mid2((l2[0]+l2[2])/2.0, (l2[1]+l2[3])/2.0);
                Point2f center((mid1.x + mid2.x)/2.0, (mid1.y + mid2.y)/2.0);

                double len1 = norm(Point2f(l1[2], l1[3]) - Point2f(l1[0], l1[1]));
                double len2 = norm(Point2f(l2[2], l2[3]) - Point2f(l2[0], l2[1]));
                double avgLength = (len1 + len2) / 2.0;

                double angleLine = line1.theta - CV_PI/2;
                angleLine = angleLine * 180 / CV_PI;
                if (angleLine < 0) angleLine += 180;

                parkingSpaces.emplace_back(center, Size2f(avgLength, diff), angleLine);
                i++;
            }
        }
    }

    return parkingSpaces;
}

Mat drawParkingSpaces(const Mat& img, const std::vector<RotatedRect>& parkingSpaces) {
    Mat out = img.clone();
    for (const auto& space : parkingSpaces) {
        Point2f vertices[4];
        space.points(vertices);
        for (int i = 0; i < 4; ++i) {
            line(out, vertices[i], vertices[(i+1)%4], Scalar(0, 255, 0), 2);
        }
    }
    return out;
}

void prova (const Mat& src) {
     // Convert to grayscale
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Optional: Preprocessing to obtain a binary image (mask)
    // You might need to adjust the threshold value based on your image.
    Mat binary;
    threshold(gray, binary, 100, 255, THRESH_BINARY_INV);

    // Optional: Clean up the binary image with morphological operations
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_OPEN, kernel);

    // Find contours in the binary image (each contour is a candidate parking space)
    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Create a copy of the original image for drawing
    Mat result = src.clone();

    // Loop over each contour
    for (size_t i = 0; i < contours.size(); i++) {
        // Skip small contours that are unlikely to be parking spaces
        if (contourArea(contours[i]) < 500) continue;

        // Compute the minimum area rotated rectangle for the current contour
        RotatedRect rotRect = minAreaRect(contours[i]);

        // Get the 4 vertices of the rotated rectangle
        Point2f vertices[4];
        rotRect.points(vertices);

        // Draw the rotated rectangle (bounding box)
        for (int j = 0; j < 4; j++) {
            line(result, vertices[j], vertices[(j + 1) % 4], Scalar(0, 0, 255), 2);
        }

        // Optionally, draw the center of the rectangle
        circle(result, rotRect.center, 5, Scalar(255, 0, 0), -1);
    }

    // Display the results
    imshow("Original Image", src);
    imshow("Binary Mask", binary);
    imshow("Rotated Bounding Boxes", result);

    waitKey(0);
}