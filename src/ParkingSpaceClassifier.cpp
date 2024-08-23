#include "ParkingSpaceClassifier.h"

using namespace cv;

// Function to create a bounding box
RotatedRect createBoundingBox(const Point2f& center, const Size2f& size, float angle) {
    return RotatedRect(center, size, angle);
}

void classifyParkingSpaces(const Mat &parkingLotImage, std::vector<RotatedRect> &parkingSpaces, std::vector<bool> &occupancyStatus) {
    for (size_t i = 0; i < parkingSpaces.size(); ++i) {
        // Get the rotation matrix for the bounding box
        const RotatedRect &rotated_rect = parkingSpaces[i];
        Mat rotation_matrix = getRotationMatrix2D(rotated_rect.center, rotated_rect.angle, 1.0);

        // Rotate the entire image around the bounding box center
        Mat rotated_image;
        warpAffine(parkingLotImage, rotated_image, rotation_matrix, parkingLotImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
        
        // Extract the ROI from the rotated image
        Size rect_size = rotated_rect.size;
        
        // Create a 3x3 full rotation matrix from the 2x3 rotation matrix
        cv::Rect bounding_box(
            static_cast<int>(rotated_rect.center.x - rect_size.width / 2),
            static_cast<int>(rotated_rect.center.y - rect_size.height / 2),
            static_cast<int>(rect_size.width),
            static_cast<int>(rect_size.height)
        );

        // Ensure the bounding box is within the image bounds
        bounding_box &= cv::Rect(0, 0, rotated_image.cols, rotated_image.rows);
        Mat cropped_image;

        if (bounding_box.width > 0 && bounding_box.height > 0) {
            cropped_image = rotated_image(bounding_box);

            // Display the cropped and scaled image
            std::string window_name = "Cropped Image " + std::to_string(&rotated_rect - &parkingSpaces[0]);
            imshow(window_name, cropped_image);
        }
        waitKey(0);  // Wait for a key press to close all windows

        // Convert to grayscale
        Mat grayRoi;
        cvtColor(cropped_image, grayRoi, COLOR_BGR2GRAY);

        // Apply Canny edge detection
        Mat edges;
        Canny(grayRoi, edges, 50, 150);
        
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
