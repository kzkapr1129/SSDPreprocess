#include <stdio.h>
#include <random>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sys/stat.h>

// ======== 前処理パラメータ
static const int SPLIT_SIZE = 2;

// ======== アプリパラメータ
static const std::string OUT_FOLDER = "out";
static const int ACCESS_FLAGS = S_IRGRP | S_IROTH | S_IRUSR | S_IRWXO | S_IRWXU | S_IWOTH;
static const std::string OUT_IMG_NAME_PREFIX = "";
static const char* OUT_FILENAME_FMT = "%llu_%d.png";

struct ImgStore {
	int index;

	void save(const cv::Mat& img) {
		char base_name[255];
		sprintf(base_name, OUT_FILENAME_FMT, time(NULL), index++);

		std::string filename = base_name;
		if (0 < OUT_IMG_NAME_PREFIX.length()) {
			filename = OUT_IMG_NAME_PREFIX + filename;
		}

		std::string path = OUT_FOLDER + "/" + filename;
		cv::imwrite(path.c_str(), img);
	}

	ImgStore() : index(0) {}
};

static void splitWimage(const cv::Mat& frame, std::vector<cv::Mat>& imgs, int paddingX, int paddingY, int numsplit) {
	cv::Size size(frame.rows-paddingY, frame.rows-paddingY);

	int addVal = ((frame.cols - frame.rows - paddingX) / numsplit);

	for (int i = 0; i + size.width + paddingX < frame.cols; i += addVal) {
		cv::Rect rect(cv::Point(i+paddingX, paddingY), size);
		cv::Mat roi = frame(rect);
		imgs.push_back(roi.clone());
	}
}

static void splitHImage(const cv::Mat& frame, std::vector<cv::Mat>& imgs, int paddingX, int paddingY, int numsplit) {
	cv::Size size(frame.cols-paddingX, frame.cols-paddingX);

	int addVal = ((frame.rows - frame.cols - paddingY) / numsplit);

	for (int i = 0; i + size.height < frame.rows; i += addVal) {
		cv::Rect rect(cv::Point(paddingX, i+paddingY), size);
		cv::Mat roi = frame(rect);
		imgs.push_back(roi.clone());
	}
}

static void splitImage(const cv::Mat& frame, std::vector<cv::Mat>& imgs, int paddingX, int paddingY, int numsplit) {
	float aspect = (float)frame.cols / (float)frame.rows;
	if (1 <= aspect) {
		splitWimage(frame, imgs, paddingX, paddingY, numsplit);
	} else {
		splitHImage(frame, imgs, paddingX, paddingY, numsplit);
	}
}

static void correctBrightness(cv::Mat& frame, float bright_scale) {
	cv::Mat hsv;
	cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

	std::vector<cv::Mat> channels;
	cv::split(hsv, channels);
	channels[2] *= bright_scale;
	cv::merge(channels, hsv);
	cv::cvtColor(hsv, frame, cv::COLOR_HSV2BGR);
}

static cv::Mat& randamRota(std::mt19937& mt, cv::Mat& frame) {
	int index = mt() % 4;
	switch(index) {
	case 0: // 90度
		cv::transpose(frame, frame);
		cv::flip(frame, frame, 1);
		break;

	case 1: // 180度
		cv::flip(frame, frame, -1);
		break;

	case 2: // 270度
		cv::transpose(frame, frame);
		cv::flip(frame, frame, 0);
		break;

	case 3: // 0度
	default:
		// IGNORE
		break;
	}

	return frame;
}

int main(int argc, char* argv[]) {
	if (argc != 6) {
		fprintf(stderr, "usage: %s video_name brightness_scale paddingx paddingy numsplit\n", argv[0]);
		return -1;
	}

	// 明度補正値
	float bright_scale = atof(argv[2]);
	if (bright_scale <= 0) {
		fprintf(stderr, "invalid brightness scale: %f\n", bright_scale);
		return -1;
	}

	int paddingX = atoi(argv[3]);
	int paddingY = atoi(argv[4]);
	int numsplit = atoi(argv[5]);

	// 動画ファイルのopen
	cv::VideoCapture cap(argv[1]);
	if (!cap.isOpened()) {
		fprintf(stderr, "couldn't open %s\n", argv[1]);
		return -1;
	}

	// 出力フォルダ作成
	mkdir(OUT_FOLDER.c_str(), ACCESS_FLAGS);

	// イメージストア初期化
	ImgStore store;

	// 乱数生成
	std::random_device rnd;
	std::mt19937 mt(rnd());

	// ビデオファイル読み込み開始
	while (true) {
		cv::Mat frame;
		cap >> frame;

		if (frame.empty()) {
			break;
		}

		// 明度補正
		correctBrightness(frame, bright_scale);

		// 正方画像になるように画像分割する
		std::vector<cv::Mat> imgs;
		splitImage(frame, imgs, paddingX, paddingY, numsplit);

		for (auto it = imgs.begin(); it != imgs.end(); it++) {
			store.save(randamRota(mt, *it));
		}
	}

	printf("ok\n");

	return 0;
}