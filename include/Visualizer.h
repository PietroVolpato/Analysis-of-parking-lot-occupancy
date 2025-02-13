#ifndef VISUALIZER_H
#define VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>


// Creates and returns a blank minimap (white background) with the specified dimensions.
cv::Mat createMinimap(int width, int height);

// Draws a single parking space (rotated rectangle) onto the provided minimap image using the specified color.
void drawParkingSpace(cv::Mat& minimap, const cv::RotatedRect& space, const cv::Scalar& color);

// Updates the minimap by drawing each parking space with its occupancy status.
// - parkingSpaces: a vector of cv::RotatedRect representing the parking space bounding boxes.
// - occupancy: a vector<bool> of the same size indicating occupancy (true for occupied, false for free).
void updateMinimap(cv::Mat& minimap, const std::vector<cv::RotatedRect>& parkingSpaces, 
                   const std::vector<bool>& occupancy);

// Displays the minimap in a window.
void showMinimap(const cv::Mat& minimap, const std::string& windowName);

void drawRotatedRect(cv::Mat& image, const cv::RotatedRect& rrect, const cv::Scalar& color);

cv::Mat createMockMinimap(int width, int height);

#endif
