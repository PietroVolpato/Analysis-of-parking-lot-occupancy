#include "ParkingSpaceClassifier.h"

using namespace cv;

// Function to create a bounding box
RotatedRect createBoundingBox(const Point2f& center, const Size2f& size, float angle) {
    return RotatedRect(center, size, angle);
}

// Function to determine if a parking space is occupied
bool isOccupied(const Mat &bbox)  {
    // Convert to grayscale for edge detection
    Mat gray_sclaed;
    cvtColor(bbox, gray_sclaed, cv::COLOR_BGR2GRAY);

    // Perform edge detection
    Mat edges;
    Canny(gray_sclaed, edges, 50, 150);

    // Count the number of edge pixels
    int edgePixelCount = countNonZero(edges);

    // Calculate the mean color of the bounding box
    Scalar meanColor = mean(bbox);

    // Threshold values need to be adjusted
    int edgeThreshold = 500;  // Edge count threshold
    double colorThreshold = 50;  // Color threshold (distance from grey)

    // Calculate the distance from asphalt color
    double distanceFromGrey = norm(meanColor - Scalar(127, 127, 127));

    // Classification logic
    if (distanceFromGrey > colorThreshold || edgePixelCount > edgeThreshold) {
        return true;  // Occupied by a car
    } else {
        return false; // Not occupied (likely just asphalt)
    }
}

// Function to classify parking spaces
void classifyParkingSpaces(const Mat &parkingLotImage, std::vector<RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        // Extract the region of interest (ROI) corresponding to the parking space
        Mat roi;
        getRectSubPix(parkingLotImage, parkingSpaces[i].size, parkingSpaces[i].center, roi);

        // Determine if the space is occupied
        occupancyStatus[i] = isOccupied(roi);
    }
}

// Function to draw the parking spaces on the image
void drawParkingSpaces(Mat &image, const std::vector<RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        Point2f vertices[4];
        parkingSpaces[i].points(vertices);
        
        Scalar color = occupancyStatus[i] ? Scalar(0, 0, 255) : Scalar(0, 255, 0); // Red for occupied, green for empty

        for (int j = 0; j < 4; j++)
            line(image, vertices[j], vertices[(j+1)%4], color, 2);
    }
}
