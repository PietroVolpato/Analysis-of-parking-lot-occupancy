#include "ParkingSpaceClassifier.h"

using namespace cv;

// Function to create a bounding box
RotatedRect createBoundingBox(const Point2f& center, const Size2f& size, float angle) {
    return RotatedRect(center, size, angle);
}

void classifyParkingSpaces(const Mat &parkingLotImage, std::vector<RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        Mat roi;
        getRectSubPix(parkingLotImage, parkingSpaces[i].size, parkingSpaces[i].center, roi);

        // Convert to grayscale
        Mat grayRoi;
        cvtColor(roi, grayRoi, COLOR_BGR2GRAY);

        // Apply Canny edge detection
        Mat edges;
        Canny(grayRoi, edges, 50, 150);
        std::string window_name = "Edge map " + std::to_string(i + 1);
        imshow(window_name, edges);
        waitKey(0);
        // Flatten the edge image for K-Means clustering
        Mat data;
        edges.convertTo(data, CV_32F);
        data = data.reshape(1, data.total());

        // Apply K-Means clustering with 2 clusters
        Mat labels, centers;
        kmeans(data, 2, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

        // Count the number of pixels in each cluster
        int occupiedCluster = centers.at<float>(0) < centers.at<float>(1) ? 0 : 1;
        int occupiedCount = countNonZero(labels == occupiedCluster);

        // Set occupancy status based on cluster with the majority of edge pixels
        occupancyStatus[i] = occupiedCount > (data.total() / 2);
    }

}

// Function to draw the parking spaces on the image
void drawParkingSpaces(Mat &image, const std::vector<RotatedRect> &parkingSpaces, const std::vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        Point2f vertices[4];
        parkingSpaces[i].points(vertices);
        
        Scalar color = occupancyStatus[i] ? Scalar(0, 0, 255) : Scalar(0, 255, 0);

        for (int j = 0; j < 4; j++)
            line(image, vertices[j], vertices[(j+1)%4], color, 2);
    }
}
