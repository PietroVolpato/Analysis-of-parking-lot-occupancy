#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cv;
using namespace std;

// Compute the angle (in degrees) of a line given by its endpoints
double computeAngle(const Vec4i &line) {
    double dx = line[2] - line[0];
    double dy = line[3] - line[1];
    double angle = atan2(dy, dx) * 180.0 / CV_PI;
    if(angle < 0) angle += 180;
    return angle;
}

// Compute a normalized line representation (a, b, c) for the line equation a*x + b*y + c = 0
// so that sqrt(a^2 + b^2) == 1.
void computeLineParameters(const Vec4i &line, float &a, float &b, float &c) {
    float dx = line[2] - line[0];
    float dy = line[3] - line[1];
    float normVal = sqrt(dx * dx + dy * dy);
    if (normVal == 0) {
        a = b = c = 0;
        return;
    }
    // We use (-dy, dx) as a normal vector (choice is arbitrary but must be consistent)
    a = -dy / normVal;
    b = dx / normVal;
    c = -(a * line[0] + b * line[1]);
}

int main(int argc, char** argv)
{
    // if(argc < 2){
    //     cout << "Usage: ./two_line_parking_detector <image_path>" << endl;
    //     return -1;
    // }
    
    // Load the input image
    Mat src = imread("/home/pietro/Analysis-of-parking-lot-occupancy/data/sequence0/frames/2013-02-24_10_05_04.jpg");
    if(src.empty()){
        cerr << "Error: Cannot load image!" << endl;
        return -1;
    }
    
    // --- Step 1: Preprocessing ---
    // Convert to grayscale, equalize histogram, and blur to reduce noise.
    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    GaussianBlur(gray, gray, Size(5, 5), 0);
    
    // --- Step 2: Edge Detection ---
    Mat edges;
    Canny(gray, edges, 50, 150, 3);
    
    // --- Step 3: Line Detection ---
    // Detect line segments using the Probabilistic Hough Transform.
    vector<Vec4i> lines;
    // Parameters: rho = 1, theta = 1° (in radians), threshold = 50, minimum line length = 30, maximum line gap = 10.
    HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 30, 10);
    
    // --- Optional Filtering: If your parking lines are known to have a certain orientation,
    // you can filter here. For example, if the lines are nearly horizontal, you might keep only
    // lines with angle close to 0° or 180°. Adjust or remove this filtering as needed.
    vector<Vec4i> filteredLines;
    for (const auto &line : lines) {
        double angle = computeAngle(line);
        // For example, if parking space boundaries are nearly horizontal:
        if (angle < 20.0 || angle > 160.0) {
            filteredLines.push_back(line);
        }
    }
    
    // For visualization, draw the filtered lines on a copy of the original image.
    Mat lineImage = src.clone();
    for (const auto &line : filteredLines) {
        cv::line(lineImage, Point(line[0], line[1]), Point(line[2], line[3]), Scalar(0, 255, 0), 2);
    }
    imshow("Filtered Lines", lineImage);
    
    // --- Step 4: Find Candidate Pairs of Parallel Lines ---
    // Since each parking space is delimited by two parallel lines,
    // we iterate over pairs of filtered lines and check:
    //   a) They are nearly parallel (angle difference within a threshold)
    //   b) Their distance is within an expected range.
    double angleThreshold = 5.0; // maximum allowed angle difference (in degrees)
    double minDistance = 20.0;   // minimum distance between lines (in pixels)
    double maxDistance = 100.0;  // maximum distance between lines (in pixels) – adjust as needed
    
    // The result image will show the candidate parking space(s)
    Mat result = src.clone();
    
    // Iterate over all pairs of filtered lines
    for (size_t i = 0; i < filteredLines.size(); i++) {
        double angle_i = computeAngle(filteredLines[i]);
        float a_i, b_i, c_i;
        computeLineParameters(filteredLines[i], a_i, b_i, c_i);
        
        for (size_t j = i+1; j < filteredLines.size(); j++) {
            double angle_j = computeAngle(filteredLines[j]);
            
            // Check that the lines are nearly parallel.
            if (fabs(angle_i - angle_j) > angleThreshold)
                continue;
            
            float a_j, b_j, c_j;
            computeLineParameters(filteredLines[j], a_j, b_j, c_j);
            
            // Compute the distance from one endpoint of line i to line j.
            // (Since the lines are parallel, the distance is (approximately) the same anywhere.)
            Point pt_i(filteredLines[i][0], filteredLines[i][1]);
            double distance = fabs(a_j * pt_i.x + b_j * pt_i.y + c_j);
            
            // Check if the distance is within an expected range.
            if (distance < minDistance || distance > maxDistance)
                continue;
            
            // Candidate pair found – combine endpoints of both lines.
            vector<Point> points;
            points.push_back(Point(filteredLines[i][0], filteredLines[i][1]));
            points.push_back(Point(filteredLines[i][2], filteredLines[i][3]));
            points.push_back(Point(filteredLines[j][0], filteredLines[j][1]));
            points.push_back(Point(filteredLines[j][2], filteredLines[j][3]));
            
            // Compute a rotated bounding box that fits these points.
            RotatedRect parkingBox = minAreaRect(points);
            Point2f rectPoints[4];
            parkingBox.points(rectPoints);
            for (int k = 0; k < 4; k++) {
                line(result, rectPoints[k], rectPoints[(k + 1) % 4], Scalar(0, 0, 255), 2);
            }
            
            // Also, draw the candidate lines (optional, for debugging)
            line(result, Point(filteredLines[i][0], filteredLines[i][1]), 
                 Point(filteredLines[i][2], filteredLines[i][3]), Scalar(255, 0, 0), 2);
            line(result, Point(filteredLines[j][0], filteredLines[j][1]), 
                 Point(filteredLines[j][2], filteredLines[j][3]), Scalar(255, 0, 0), 2);
        }
    }
    
    // --- Step 5: Display the Results ---
    imshow("Detected Parking Spaces", result);
    waitKey(0);
    
    return 0;
}
