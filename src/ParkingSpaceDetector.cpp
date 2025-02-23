// PIETRO VOLPATO

#include "ParkingSpaceDetector.h"

using namespace cv;
using namespace std;

//-------------------------------------------------------------
// Compute line parameters (angle, endpoints, and length) for each detected line.
vector<LineParams> ParkingSpaceDetector::computeLineParams(const vector<Vec4i>& lines) {
    vector<LineParams> params;
    for (const auto& line : lines) {
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        double a = y2 - y1;
        double b = x2 - x1;
        // Compute angle in degrees and normalize to [0, 180)
        double angle = atan2(a, b) * 180.0 / CV_PI;
        angle = fmod(angle, 180.0);
        if (angle < 0)
            angle += 180.0;
        double length = norm(Point(x2, y2) - Point(x1, y1));
        params.push_back({angle, line, length});
    }
    return params;
}

//-------------------------------------------------------------
// Apply region-of-interest (ROI) masking by excluding defined areas.
// This version creates a single mask that removes unwanted regions.
Mat ParkingSpaceDetector::applyRoi(const Mat& img) {
    // Create a white mask (all pixels allowed)
    Mat mask = Mat::ones(img.size(), CV_8UC1) * 255;
    
    // Define the first region to remove (set to black)
    vector<Point> roi1 = {
        Point(0, 0),
        Point(static_cast<int>(img.cols * 0.15), 0),
        Point(static_cast<int>(img.cols * 0.47), img.rows),
        Point(0, img.rows)
    };
    
    // Define the second region to remove
    vector<Point> roi2 = {
        Point(static_cast<int>(img.cols * 0.65), 0),
        Point(img.cols, 0),
        Point(img.cols, static_cast<int>(img.rows * 0.4)),
        Point(static_cast<int>(img.cols * 0.65), 0)
    };
    
    // Fill the defined ROIs with black (mask out)
    fillPoly(mask, vector<vector<Point>>{roi1, roi2}, Scalar(0));
    
    Mat masked;
    // Convert mask to 3 channels if the image is colored.
    if (img.channels() == 3) {
        Mat maskColor;
        cvtColor(mask, maskColor, COLOR_GRAY2BGR);
        bitwise_and(img, maskColor, masked);
    } else {
        bitwise_and(img, mask, masked);
    }
    return masked;
}

//-------------------------------------------------------------
// Perform CLAHE-based equalization.
Mat ParkingSpaceDetector::equalization(const Mat& img) {
    Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8, 8));
    Mat out;
    clahe->apply(img, out);
    return out;
}

//-------------------------------------------------------------
// Apply gamma correction using a lookup table.
Mat ParkingSpaceDetector::gammaCorrection(const Mat& img, const double gamma) {
    Mat lookUpTable(1, 256, CV_8U);
    uchar* p = lookUpTable.ptr();
    for (int i = 0; i < 256; ++i) {
        p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    Mat correctedImg;
    LUT(img, lookUpTable, correctedImg);
    return correctedImg;
}

//-------------------------------------------------------------
// Check if two angles represent parallel lines (within tolerance).
bool ParkingSpaceDetector::isParallel(const double theta1, const double theta2) {
    const double tolerance = 5.0;
    double diff = fabs(fabs(theta1) - fabs(theta2));
    diff = min(diff, 180.0 - diff);
    return diff <= tolerance;
}

//-------------------------------------------------------------
// Calculate the minimum distance between two lines.
double ParkingSpaceDetector::distanceBetweenLines(const Vec4i& line1, const Vec4i& line2) {
    Point2f p1(line1[0], line1[1]), q1(line1[2], line1[3]);
    Point2f p2(line2[0], line2[1]), q2(line2[2], line2[3]);
    
    auto pointSegmentDistance = [](const Point2f& p, const Point2f& a, const Point2f& b) {
        Point2f ab = b - a, ap = p - a;
        float t = ap.dot(ab) / ab.dot(ab);
        t = max(0.0f, min(1.0f, t));
        Point2f closest = a + t * ab;
        return norm(p - closest);
    };
    
    double d1 = pointSegmentDistance(p1, p2, q2);
    double d2 = pointSegmentDistance(q1, p2, q2);
    double d3 = pointSegmentDistance(p2, p1, q1);
    double d4 = pointSegmentDistance(q2, p1, q1);
    
    return min({d1, d2, d3, d4});
}

//-------------------------------------------------------------
// Merge a group of close lines by selecting the pair of endpoints with the maximum separation.
Vec4i ParkingSpaceDetector::mergeCloseLines(const vector<Vec4i>& closeLines) {
    if (closeLines.empty())
        return Vec4i(0, 0, 0, 0);
    
    vector<Point> endpoints;
    for (const auto& line : closeLines) {
        endpoints.push_back(Point(line[0], line[1]));
        endpoints.push_back(Point(line[2], line[3]));
    }
    
    // Check if all endpoints are identical.
    bool allSame = all_of(endpoints.begin(), endpoints.end(), 
                          [&endpoints](const Point& p) { return p == endpoints[0]; });
    if (allSame)
        return Vec4i(endpoints[0].x, endpoints[0].y, endpoints[0].x, endpoints[0].y);
    
    double maxDistSquared = 0;
    pair<Point, Point> farthest;
    for (size_t i = 0; i < endpoints.size(); ++i) {
        for (size_t j = i + 1; j < endpoints.size(); ++j) {
            double dx = endpoints[i].x - endpoints[j].x;
            double dy = endpoints[i].y - endpoints[j].y;
            double distSquared = dx * dx + dy * dy;
            if (distSquared > maxDistSquared) {
                maxDistSquared = distSquared;
                farthest = {endpoints[i], endpoints[j]};
            }
        }
    }
    return Vec4i(farthest.first.x, farthest.first.y, farthest.second.x, farthest.second.y);
}

//-------------------------------------------------------------
// Find lines that are close to a reference line.
vector<Vec4i> ParkingSpaceDetector::findCloseLines(const Vec4i& reference, const vector<LineParams>& lines) {
    vector<Vec4i> closeLines;
    for (const auto& line : lines) {
        if (distanceBetweenLines(reference, line.endpoints) < 15.0)
            closeLines.push_back(line.endpoints);
    }
    return closeLines;
}

//-------------------------------------------------------------
// Merge nearby lines into a single representative line.
vector<Vec4i> ParkingSpaceDetector::mergeLines(const vector<LineParams>& lines) {
    vector<Vec4i> mergedLines;
    vector<bool> used(lines.size(), false);
    
    for (size_t i = 0; i < lines.size(); ++i) {
        if (used[i])
            continue;
        vector<Vec4i> closeLines { lines[i].endpoints };
        for (size_t j = i + 1; j < lines.size(); ++j) {
            if (used[j])
                continue;
            if (distanceBetweenLines(lines[i].endpoints, lines[j].endpoints) < 15.0 &&
                isParallel(lines[i].angle, lines[j].angle)) {
                closeLines.push_back(lines[j].endpoints);
                used[j] = true;
            }
        }
        mergedLines.push_back(mergeCloseLines(closeLines));
    }
    return mergedLines;
}

//-------------------------------------------------------------
// Create a bounding box (RotatedRect) from two lines.
RotatedRect ParkingSpaceDetector::createBoundingBox(const LineParams& line1, const LineParams& line2) {
    Point2f pts[4] = {
        Point2f(line1.endpoints[0], line1.endpoints[1]),
        Point2f(line1.endpoints[2], line1.endpoints[3]),
        Point2f(line2.endpoints[0], line2.endpoints[1]),
        Point2f(line2.endpoints[2], line2.endpoints[3])
    };
    vector<Point2f> points(pts, pts + 4);
    return minAreaRect(points);
}

//-------------------------------------------------------------
// Preprocess the image by converting to grayscale, enhancing contrast,
// applying gamma correction, and masking unwanted regions.
Mat ParkingSpaceDetector::preprocessImage(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    // Apply CLAHE equalization and gamma correction (choose one or combine)
    Mat equalized = equalization(gray);
    Mat gammaCorrected = gammaCorrection(gray, 1.5);
    
    // Here, we use gamma-corrected image; you can experiment with equalized as well.
    Mat masked = applyRoi(gray);
    
    return masked;
}

//-------------------------------------------------------------
// Detect edges using the Canny algorithm.
Mat ParkingSpaceDetector::detectEdges(const Mat& img, int threshold1, int threshold2, int aperture) {
    Mat edges;
    Canny(img, edges, threshold1, threshold2, aperture);
    return edges;
}

//-------------------------------------------------------------
// Display an image in a window.
void ParkingSpaceDetector::showImage(const Mat& img) {
    imshow("Image", img);
    waitKey(0);
}

//-------------------------------------------------------------
// Detect lines using OpenCV's LineSegmentDetector.
vector<Vec4i> ParkingSpaceDetector::detectLines(const Mat& img) {
    vector<Vec4i> lines;
    Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(LSD_REFINE_NONE);
    vector<Vec4f> linesFloat;
    lsd->detect(img, linesFloat);
    for (const auto& line : linesFloat) {
        lines.push_back(Vec4i(cvRound(line[0]), cvRound(line[1]), cvRound(line[2]), cvRound(line[3])));
    }
    return lines;
}

//-------------------------------------------------------------
// Draw detected lines on the image.
void ParkingSpaceDetector::drawLines(Mat& img, const vector<LineParams>& lines) {
    for (const auto& line : lines) {
        Point p1(line.endpoints[0], line.endpoints[1]);
        Point p2(line.endpoints[2], line.endpoints[3]);
        cv::line(img, p1, p2, Scalar(0, 0, 255), 2);
    }
}

//-------------------------------------------------------------
// Filter and refine lines using angle, length, and a local PCA for outlier removal.
vector<LineParams> ParkingSpaceDetector::filterLines(vector<LineParams>& lines) {
    vector<LineParams> filteredLines;
    const double angleTarget1 = 15.0;
    const double angleTarget2Low = 90.0;
    const double angleTarget2High = 125.0;
    const double tolerance = 10.0;
    
    // Initial filtering based on angle and length using tolerance for both targets.
    for (const auto& line : lines) {
        double angle = line.angle;
        if ((fabs(angle - angleTarget1) < tolerance) || (angle > angleTarget2Low && angle < angleTarget2High) &&
             line.length > 25.0) {
            filteredLines.push_back(line);
        }
    }

    // Refine using local PCA to remove lines with inconsistent orientation.
    vector<LineParams> refinedLines;
    for (size_t i = 0; i < filteredLines.size(); ++i) {
        vector<Vec4i> closeLineEndpoints = findCloseLines(filteredLines[i].endpoints, filteredLines);
        vector<LineParams> closeLines = computeLineParams(closeLineEndpoints);
        
        vector<Point2f> points;
        for (const auto& cl : closeLines) {
            points.push_back(Point2f(cl.endpoints[0], cl.endpoints[1]));
            points.push_back(Point2f(cl.endpoints[2], cl.endpoints[3]));
        }
        if (points.empty()) continue;
        
        Mat data(points.size(), 2, CV_32F);
        for (size_t j = 0; j < points.size(); ++j) {
            data.at<float>(j, 0) = points[j].x;
            data.at<float>(j, 1) = points[j].y;
        }
        PCA pca(data, Mat(), PCA::DATA_AS_ROW);
        Point2f dir(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));
        double mainAngle = atan2(dir.y, dir.x) * 180.0 / CV_PI;
        if (fabs(mainAngle - filteredLines[i].angle) < 20.0)
            refinedLines.push_back(filteredLines[i]);
    }
    
    // Merge nearby lines and recompute parameters.
    vector<Vec4i> merged = mergeLines(refinedLines);

    return computeLineParams(merged);
}

//-------------------------------------------------------------
// Cluster lines based on their angles using a simple k-means approach.
// pair<vector<LineParams>, vector<LineParams>> ParkingSpaceDetector::clusterLines(const vector<LineParams>& lines) {
//     if (lines.empty())
//         return {{}, {}};
    
//     // double cluster1Mean = 15.0;
//     // double cluster2Mean = 105.0;
//     // vector<LineParams> cluster1, cluster2;
//     // bool converged = false;
//     // int iterations = 0;
    
//     // while (!converged && iterations < 100) {
//     //     vector<LineParams> newCluster1, newCluster2;
//     //     for (const auto& line : lines) {
//     //         double dist1 = fabs(line.angle - cluster1Mean);
//     //         double dist2 = fabs(line.angle - cluster2Mean);
//     //         if (dist1 < dist2)
//     //             newCluster1.push_back(line);
//     //         else
//     //             newCluster2.push_back(line);
//     //     }
        
//     //     double newCluster1Mean = newCluster1.empty() ? cluster1Mean : 0.0;
//     //     for (const auto& line : newCluster1)
//     //         newCluster1Mean += line.angle;
//     //     if (!newCluster1.empty())
//     //         newCluster1Mean /= newCluster1.size();
        
//     //     double newCluster2Mean = newCluster2.empty() ? cluster2Mean : 0.0;
//     //     for (const auto& line : newCluster2)
//     //         newCluster2Mean += line.angle;
//     //     if (!newCluster2.empty())
//     //         newCluster2Mean /= newCluster2.size();
        
//     //     if (fabs(newCluster1Mean - cluster1Mean) < 0.1 && fabs(newCluster2Mean - cluster2Mean) < 0.1)
//     //         converged = true;
        
//     //     cluster1Mean = newCluster1Mean;
//     //     cluster2Mean = newCluster2Mean;
//     //     cluster1 = newCluster1;
//     //     cluster2 = newCluster2;
//     //     iterations++;
//     // }
    
//     // return {cluster1, cluster2};
//     vector<LineParams> cluster1, cluster2;
//     for (const auto& line: lines) {
//         if (fabs(line.angle - 15.0) < 10.0)
//             cluster1.push_back(line);
//         else
//             cluster2.push_back(line);
//     }

//     return {cluster1, cluster2};
// }

//-------------------------------------------------------------
// Cluster lines based on their angles
pair<vector<pair<LineParams, LineParams>>, vector<pair<LineParams, LineParams>>> ParkingSpaceDetector::clusterLines(const vector<LineParams>& lines) {
    if (lines.empty())
        return {{}, {}};
    
    vector<pair<LineParams, LineParams>> cluster1, cluster2;
    for (size_t i = 0; i < lines.size(); ++i) {
        for (size_t j = i + 1; j < lines.size(); ++j) {
            if (fabs(lines[i].angle - 15.0) < 10.0 && fabs(lines[j].angle - 15.0) < 10.0)
                cluster1.push_back({lines[i], lines[j]});
            else if (fabs(lines[i].angle - 105.0) < 10.0 && fabs(lines[j].angle - 105.0) < 10.0)
                cluster2.push_back({lines[i], lines[j]});
        }
    }
    return {cluster1, cluster2};
}

//-------------------------------------------------------------
// Detect parking spaces by pairing parallel lines from each cluster,
// creating bounding boxes, and filtering by geometric criteria.
vector<RotatedRect> ParkingSpaceDetector::detectParkingSpaces(const pair<vector<pair<LineParams, LineParams>>, vector<pair<LineParams, LineParams>>>& clusteredLines) {
    vector<RotatedRect> parkingSpaces;
    
    // Lambda per processare un cluster di coppie di linee
    auto processClusterPairs = [&](const vector<pair<LineParams, LineParams>>& clusterPairs) {
        for (const auto& linePair : clusterPairs) {
            // Verifica che le due linee siano parallele
            if (isParallel(linePair.first.angle, linePair.second.angle)) {
                double dist = distanceBetweenLines(linePair.first.endpoints, linePair.second.endpoints);
                if (dist > 40.0 && dist < 100.0) {
                    parkingSpaces.push_back(createBoundingBox(linePair.first, linePair.second));
                }
            }
        }
    };
    
    processClusterPairs(clusteredLines.first);
    processClusterPairs(clusteredLines.second);
    
    vector<RotatedRect> filtered;
    const double angleThreshold1 = 15.0;
    const double angleThreshold2Low = 90.0;
    const double angleThreshold2High = 125.0;
    const double angleTolerance = 10.0;
    
    // Not working at all
    for (const auto& rect : parkingSpaces) {
        double angle = fmod(rect.angle, 180.0);
        if (angle < 0)
            angle += 180.0;
        
            if (((fabs(angle - angleThreshold1) < 10.0) ||
            (angle > angleThreshold2Low && angle < angleThreshold2High)) && rect.size.area() > 10000 && rect.size.area() < 20000) {
            filtered.push_back(rect);
        }
    }
    
    return filtered;
}

//-------------------------------------------------------------
// Detect parking spaces by pairing parallel lines and filtering by distance.
vector<RotatedRect> ParkingSpaceDetector::detectParkingSpacesSimple (const vector<LineParams>& lines) {
    vector<RotatedRect> parkingSpaces;
    for (size_t i = 0; i < lines.size(); i++) {
        for (size_t j = i + 1; j < lines.size(); j++) {
            if (isParallel(lines[i].angle, lines[j].angle)) {
                double dist = distanceBetweenLines(lines[i].endpoints, lines[j].endpoints);
                if (dist > 40.0 && dist < 100.0) {
                    parkingSpaces.push_back(createBoundingBox(lines[i], lines[j]));
                }
            }
        }
    }

    // Filter the parking spaces based on angle and distance
    vector<RotatedRect> filtered;
    const double angleThreshold1 = 15.0;
    const double angleThreshold2Low = 90.0;
    const double angleThreshold2High = 125.0;

    for (const auto& rect : parkingSpaces) {
        double angle = fmod(rect.angle, 180.0);
        if (angle < 0)
            angle += 180.0;

        if (((fabs(angle - angleThreshold1) < 10.0) ||
            (angle > angleThreshold2Low && angle < angleThreshold2High)) && rect.size.area() > 500 && rect.size.area() < 1000) {
            filtered.push_back(rect);
        }
    }

    return filtered;
}

//-------------------------------------------------------------
// Draw parking spaces (RotatedRect) on the image.
Mat ParkingSpaceDetector::drawParkingSpaces(const Mat& img, const vector<RotatedRect>& parkingSpaces) {
    Mat out = img.clone();
    for (const auto& space : parkingSpaces) {
        Point2f vertices[4];
        space.points(vertices);
        for (int i = 0; i < 4; ++i) {
            line(out, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
        }
    }
    return out;
}
