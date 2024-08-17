#include <ParkingSpaceClassifier.h>

using namespace cv;

// Function to create a bounding box
RotatedRect createBoundingBox(const Point2f& center, const Size2f& size, float angle) {
    return RotatedRect(center, size, angle);
}

// Function to determine if a parking space is occupied
bool isOccupied(const Mat &roi) {
    // Convert ROI to grayscale
    Mat grayRoi;
    cvtColor(roi, grayRoi, COLOR_BGR2GRAY);

    // Thresholding to create a binary image
    Mat binaryRoi;
    threshold(grayRoi, binaryRoi, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // Count non-zero pixels (i.e., white pixels) in the binary image
    int nonZeroCount = countNonZero(binaryRoi);

    // Heuristic: If there are many non-zero pixels, the space is occupied
    return nonZeroCount > 0.2 * binaryRoi.total(); // Adjust the threshold as necessary
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
