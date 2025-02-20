#include "ParkingSpaceDetector.h"

using namespace cv;
using namespace std;

vector<Mat> ParkingSpaceDetector::loadImages(const int sequence) {
    vector<String> fileNames;
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

    vector<Mat> imgs;
    for (const auto& file : fileNames) {
        Mat img = imread(file);
        if (img.empty()) {
            cerr << "Error loading image: " << file << endl;
            continue;
        }
        imgs.push_back(img);
    }
    return imgs;
}

vector<LineParams> ParkingSpaceDetector::computeLineParams(const vector<Vec4i>& lines) {
    vector<LineParams> params;
    for (const auto& line : lines) {
        int x1 = line[0], y1 = line[1], x2 = line[2], y2 = line[3];
        double a = y2 - y1;
        double b = x2 - x1;
        double c = x1 * y2 - x2 * y1;
        double angle = atan2(a, b) * 180 / CV_PI; // angle in degrees
        // Normalize the angle to be in the range [0, 180]
        angle = fmod(angle, 180);
        if (angle < 0) angle += 180;
        double length = sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)); 
        params.push_back({angle, line, length});
    }
    return params;
}

Mat ParkingSpaceDetector::applyRoi(const Mat& img) {
    Mat mask = Mat::zeros(img.size(), img.type());
    vector<vector<Point>> roi;
    Point pts[4] = {
        Point(0, 0),
        Point(img.cols * 0.15, 0),
        Point(img.cols * 0.47 , img.rows),
        Point(0, img.rows),
    };
    roi.push_back(vector<Point>(pts, pts + 4));
    fillPoly(mask, roi, Scalar(255));

    bitwise_not(mask, mask);

    Mat masked;
    bitwise_and(img, mask, masked);

    roi.clear();
    mask = Mat::zeros(img.size(), img.type());

    Point pts2[4] = {
        Point(img.cols * 0.65, 0),
        Point(img.cols, 0),
        Point(img.cols, img.rows * 0.4),
        Point(img.cols * 0.65, 0)
    };
    roi.push_back(vector<Point>(pts2, pts2 + 4));
    fillPoly(mask, roi, Scalar(255));

    bitwise_not(mask, mask);
    bitwise_and(masked, mask, masked);
    return masked;
}

Mat ParkingSpaceDetector::equalization(const Mat& img) {
    Ptr<CLAHE> clahe = cv::createCLAHE(2.0, Size(8, 8));
    Mat out;
    clahe->apply(img, out);

    return out;
}

// double ParkingSpaceDetector::distanceBetweenLines(const Vec4i& line1, const Vec4i& line2) {
//     float mx1 = (line1[0] + line1[2]) / 2.0;
//     float my1 = (line1[1] + line1[3]) / 2.0;
//     float mx2 = (line2[0] + line2[2]) / 2.0;
//     float my2 = (line2[1] + line2[3]) / 2.0;

//     return hypot(mx2 - mx1, my2 - my1);

// }

bool ParkingSpaceDetector::isParallel(const double theta1, const double theta2) {
    double tollerance = 5;
    double diff = abs(abs(theta1) - abs(theta2));
    diff = min(diff, 180 - diff);

    return diff <= tollerance;
}

double ParkingSpaceDetector::distanceBetweenLines (const Vec4i& line1, const Vec4i& line2) {
    Point2f p1(line1[0], line1[1]), q1(line1[2], line1[3]);
    Point2f p2(line2[0], line2[1]), q2(line2[2], line2[3]);

    // Function to calculate the distance between a point and a segment
    auto pointSegmentDistance = [](const Point2f& p, const Point2f& a, const Point2f& b) {
        Point2f ab = b - a, ap = p - a;
        float t = ap.dot(ab) / ab.dot(ab);
        t = max(0.0f, min(1.0f, t)); // Limita t tra 0 e 1 per rimanere nel segmento
        Point2f closest = a + t * ab;
        return norm(p - closest);
    };

    // Calculate the distance between the endpoints of the two lines
    double d1 = pointSegmentDistance(p1, p2, q2);
    double d2 = pointSegmentDistance(q1, p2, q2);
    double d3 = pointSegmentDistance(p2, p1, q1);
    double d4 = pointSegmentDistance(q2, p1, q1);

    return min({d1, d2, d3, d4});
}

// Vec4i ParkingSpaceDetector::mergeCloseLines (const vector<Vec4i>& closeLines) {
//     // Collect all endpoints from each line.
//     vector<Point> endpoints;
//     for (const auto& line : closeLines) {
//         endpoints.push_back(Point(line[0], line[1]));
//         endpoints.push_back(Point(line[2], line[3]));
//     }

//     // Find the two endpoints that are farthest apart.
//     double maxDist = 0;
//     pair<Point, Point> farthest;
//     for (size_t i = 0; i < endpoints.size(); ++i) {
//         for (size_t j = i + 1; j < endpoints.size(); ++j) {
//             double dist = norm(endpoints[i] - endpoints[j]);
//             if (dist > maxDist) {
//                 maxDist = dist;
//                 farthest = {endpoints[i], endpoints[j]};
//             }
//         }
//     }

//     return Vec4i(farthest.first.x, farthest.first.y, farthest.second.x, farthest.second.y);
// }

Vec4i ParkingSpaceDetector::mergeCloseLines(const vector<Vec4i>& closeLines) {
    // Handle edge cases
    if (closeLines.empty()) {
        return Vec4i(0, 0, 0, 0); // Return an invalid line
    }

    // Collect all endpoints from each line
    vector<Point> endpoints;
    for (const auto& line : closeLines) {
        endpoints.push_back(Point(line[0], line[1]));
        endpoints.push_back(Point(line[2], line[3]));
    }

    // Handle degenerate case (all endpoints are the same)
    bool allSame = true;
    Point firstPoint = endpoints[0];
    for (const auto& point : endpoints) {
        if (point != firstPoint) {
            allSame = false;
            break;
        }
    }
    if (allSame) {
        return Vec4i(firstPoint.x, firstPoint.y, firstPoint.x, firstPoint.y); // Degenerate line
    }

    // Find the two endpoints that are farthest apart (using squared distance for efficiency)
    double maxDistSquared = 0;
    pair<Point, Point> farthest;
    for (size_t i = 0; i < endpoints.size(); ++i) {
        for (size_t j = i + 1; j < endpoints.size(); ++j) {
            double dx = endpoints[i].x - endpoints[j].x;
            double dy = endpoints[i].y - endpoints[j].y;
            double distSquared = dx * dx + dy * dy; // Squared distance
            if (distSquared > maxDistSquared) {
                maxDistSquared = distSquared;
                farthest = {endpoints[i], endpoints[j]};
            }
        }
    }

    // Return the merged line defined by the farthest endpoints
    return Vec4i(farthest.first.x, farthest.first.y, farthest.second.x, farthest.second.y);
}

vector<Vec4i> ParkingSpaceDetector::findCloseLines (const Vec4i& reference, const vector<LineParams>& lines) {
    vector<Vec4i> closeLines;
    for (const auto& line : lines) {
        if (distanceBetweenLines(reference, line.endpoints) < 15) {
            closeLines.push_back(line.endpoints);
        }
    }
    return closeLines;
}

// Vec4i ParkingSpaceDetector::mergeCloseLines(const vector<Vec4i>& closeLines) {
//     if (closeLines.empty()) {
//         return Vec4i(0, 0, 0, 0);
//     }

//     // Collect all endpoints from each line
//     vector<Point2f> points;
//     for (const auto& line : closeLines) {
//         points.emplace_back(line[0], line[1]);
//         points.emplace_back(line[2], line[3]);
//     }

//     // Handle degenerate case (only one point)
//     if (points.size() == 1) {
//         return Vec4i(points[0].x, points[0].y, points[0].x, points[0].y);
//     }

//     // Compute the centroid of all points
//     Point2f centroid(0, 0);
//     for (const auto& p : points) {
//         centroid += p;
//     }
//     centroid.x /= points.size();
//     centroid.y /= points.size();

//     // Use PCA for finding the principal direction
//     Mat data(points.size(), 2, CV_32F);
//     for (size_t i = 0; i < points.size(); i++) {
//         data.at<float>(i, 0) = points[i].x - centroid.x;
//         data.at<float>(i, 1) = points[i].y - centroid.y;
//     }

//     PCA pca(data, Mat(), PCA::DATA_AS_ROW);
//     Point2f dir(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));

//     // Find the two endpoints that are farthest along the principal direction
//     float minProj = numeric_limits<float>::max();
//     float maxProj = numeric_limits<float>::lowest();
//     Point2f minPoint, maxPoint;

//     for (const auto& p : points) {
//         float proj = (p - centroid).dot(dir);
//         if (proj < minProj) {
//             minProj = proj;
//             minPoint = p;
//         }
//         if (proj > maxProj) {
//             maxProj = proj;
//             maxPoint = p;
//         }
//     }

//     return Vec4i(minPoint.x, minPoint.y, maxPoint.x, maxPoint.y);
// }


vector<Vec4i> ParkingSpaceDetector::mergeLines(const vector<LineParams>& lines) {
    vector<Vec4i> mergedLines;
    vector<bool> used(lines.size(), false);

    for (size_t i = 0; i < lines.size(); i++) {
        if (used[i]) continue;
        vector<Vec4i> closeLines;
        closeLines.push_back(lines[i].endpoints);
        for (size_t j = i + 1; j < lines.size(); j++) {
            if (used[j]) continue;
            if (distanceBetweenLines(lines[i].endpoints, lines[j].endpoints) < 15) {
                closeLines.push_back(lines[j].endpoints);
                used[j] = true;
            }
        }
        mergedLines.push_back(mergeCloseLines(closeLines));
        closeLines.clear();
    }
    return mergedLines;
}

RotatedRect ParkingSpaceDetector::createBoundingBox (const LineParams& line1, const LineParams& line2) {
    Point2f vertices[4];
    Point p1(line1.endpoints[0], line1.endpoints[1]);
    Point p2(line1.endpoints[2], line1.endpoints[3]);
    Point p3(line2.endpoints[0], line2.endpoints[1]);
    Point p4(line2.endpoints[2], line2.endpoints[3]);
    vertices[0] = p1;
    vertices[1] = p2;
    vertices[2] = p3;
    vertices[3] = p4;
    vector<Point2f> pts(vertices, vertices + 4);
    RotatedRect rect = minAreaRect(pts);
    
    return rect;
}

Mat ParkingSpaceDetector::preprocessImage(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat equalized = equalization(gray);
    Mat blurred;
    // GaussianBlur(equalized, blurred, Size(3, 3), 1.5);
    bilateralFilter(gray, blurred, 3, 75, 75);
    
    Mat masked = applyRoi(blurred);

    return masked;
}

Mat ParkingSpaceDetector::detectEdges(const Mat& img, int threshold1, int threshold2, int aperture) {
    Mat edges;
    Canny(img, edges, threshold1, threshold2, aperture);
    return edges;
}

void ParkingSpaceDetector::showImage(const Mat& img) {
    imshow("Image", img);
    waitKey(0);
}

vector<Vec4i> ParkingSpaceDetector::detectLines(const Mat& img, int threshold, double minLineLength, double maxLineGap) {
    vector<Vec4i> lines;
    HoughLinesP(img, lines, 1, CV_PI/180, threshold, minLineLength, maxLineGap);
    // Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(LSD_REFINE_ADV, 0.8, 0.1, 2.0, 22.5, 0, 0.9, 1024);
    // Ptr<LineSegmentDetector> lsd = createLineSegmentDetector(LSD_REFINE_ADV);
    // std::vector<Vec4f> lines_std;
    // lsd->detect(img, lines_std);
    // for (const auto& line : lines_std) {
    //     lines.push_back(Vec4i(line[0], line[1], line[2], line[3]));
    // }

    return lines;
}

void ParkingSpaceDetector::drawLines(Mat& img, const vector<LineParams>& lines) {
    for (const auto& line : lines) {
        Point p1(line.endpoints[0], line.endpoints[1]);
        Point p2(line.endpoints[2], line.endpoints[3]);
        cv::line(img, p1, p2, Scalar(0, 0, 255), 2);
    }
}

vector<LineParams> ParkingSpaceDetector::filterLines(vector<LineParams>& lines) {
    vector<LineParams> filteredLines;
    double angleTarget_1 = 15;
    double angleTarget_2_lw = 90;
    double angleTarget_2_up = 125;
    double tollerance = 10;
    for (const auto& line: lines) {
        double angle = line.angle;
        double length = line.length;
        if ((fabs(angle - angleTarget_1) < tollerance || (angle > angleTarget_2_lw && angle < angleTarget_2_up)) && length > 25) {
            filteredLines.push_back(line);
        }
    }

    // Analyze the lines for deleting the ones that have a disfference orientation from the majority
    vector<LineParams> filteredLines2;
    for (size_t i = 0; i < filteredLines.size(); i ++) {
        // Find the close lines to the current line
        vector<LineParams> closeLines = computeLineParams(findCloseLines(filteredLines[i].endpoints, filteredLines));
        // Find the majority orientation using PCA
        vector<Point2f> points;
        for (const auto& line: closeLines) {
            points.emplace_back(line.endpoints[0], line.endpoints[1]);
            points.emplace_back(line.endpoints[2], line.endpoints[3]);
        }
        Mat data(points.size(), 2, CV_32F);
        for (size_t i = 0; i < points.size(); i++) {
            data.at<float>(i, 0) = points[i].x;
            data.at<float>(i, 1) = points[i].y;
        }
        PCA pca(data, Mat(), PCA::DATA_AS_ROW);
        Point2f dir(pca.eigenvectors.at<float>(0, 0), pca.eigenvectors.at<float>(0, 1));

        double mainAngle = atan2(dir.y, dir.x) * 180 / CV_PI;
        // mainAngle= fmod(mainAngle, 180);
        // if (mainAngle < 0) mainAngle += 180;

        if (fabs(mainAngle - filteredLines[i].angle) < 20) {
            filteredLines2.push_back(filteredLines[i]);
        }
    }

    vector<Vec4i> mergedLines = mergeLines(filteredLines2);
    
    return computeLineParams(mergedLines);

    // return filteredLines2;
}

pair<vector<LineParams>, vector<LineParams>> ParkingSpaceDetector::clusterLines(const vector<LineParams>& lines) {
    vector<LineParams> cluster1; // For lines ~15 degrees
    vector<LineParams> cluster2; // For lines between 90 and 125 degrees
    
    double angleTarget_1 = 15;
    double angleTarget_2_lw = 90;
    double angleTarget_2_up = 125;
    double tolerance = 10;
    
    for (const auto& line: lines) {
        double angle = line.angle;
        if (fabs(angle - angleTarget_1) < tolerance) {
            cluster1.push_back(line);
        } else if (angle > angleTarget_2_lw && angle < angleTarget_2_up) {
            cluster2.push_back(line);
        }
    }
    
    return {cluster1, cluster2};
}

vector<RotatedRect> ParkingSpaceDetector::detectParkingSpaces(const pair<vector<LineParams>, vector<LineParams>>& cluteredLines) {
    vector<RotatedRect> parkingSpaces;
    // From the clustered lines, create clusters of dimension 2 based on the angle of the lines 
    vector<LineParams> cluster1 = cluteredLines.first;
    vector<LineParams> cluster2 = cluteredLines.second;

    for (size_t i = 0; i < cluster1.size(); i++) {
        for (size_t j = i + 1; j < cluster1.size(); j++) {
            double angle1 = cluster1[i].angle;
            double angle2 = cluster1[j].angle;
            // if (isParallel(angle1, angle2) && 
               if (distanceBetweenLines(cluster1[i].endpoints, cluster1[j].endpoints) > 40 && distanceBetweenLines(cluster1[i].endpoints, cluster1[j].endpoints) < 100) {
                RotatedRect rect = createBoundingBox(cluster1[i], cluster1[j]);
                parkingSpaces.push_back(rect);
            }
        }
    }

    const double minAspectRatio = 1.5;
    const double maxAspectRatio = 3.0;
    const double minArea = 1000;
    const double maxArea = 10000;
    const double angleThreshold_1 = 15;
    const double angleThreshold_2_lw = 90;
    const double angleThreshold_2_up = 125;
    const double angleTolerance = 10;

    vector<RotatedRect> filtered;
    for (const auto& rect : parkingSpaces) {
        double ratio = rect.size.width / rect.size.height;
        double area = rect.size.width * rect.size.height;
        double angle = rect.angle;
        // Normalize the angle to be in the range [0, 180]
        angle = fmod(angle, 180);
        if (angle < 0) angle += 180;
        
        // if (ratio > minAspectRatio && ratio < maxAspectRatio &&
        //   if(  area > minArea && area < maxArea &&
           if ((fabs(angle - angleThreshold_1) < angleTolerance) || (angle > angleThreshold_2_lw && angle < angleThreshold_2_up)) {
            filtered.push_back(rect);
        }
    }
    parkingSpaces = filtered;


    return parkingSpaces;
}

Mat ParkingSpaceDetector::drawParkingSpaces(const Mat& img, const vector<RotatedRect>& parkingSpaces) {
    Mat out = img.clone();
    for (const auto& space : parkingSpaces) {
        Point2f vertices[4];
        space.points(vertices);
        for (int i = 0; i < 4; ++i) {
            cv::line(out, vertices[i], vertices[(i+1)%4], Scalar(0, 255, 0), 2);
        }
    }
    return out;
}