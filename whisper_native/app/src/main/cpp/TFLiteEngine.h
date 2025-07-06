#ifndef _TFLITEENGINE_H_
#define _TFLITEENGINE_H_

#include <string>
#include <vector>
#include "tensorflow/lite/delegates/gpu/delegate.h"

class TFLiteEngine {
public:
    TFLiteEngine() : gpu_delegate_(nullptr) {};
    ~TFLiteEngine() {};

    int loadModel(const char *modelPath, const bool isMultilingual);
    void freeModel();

    std::string transcribeBuffer(std::vector<float> samples);
    std::string transcribeFile(const char* waveFile);

private:
    TfLiteDelegate* gpu_delegate_;
    bool initializeGpuDelegate();
    void cleanupGpuDelegate();
};

#endif // _TFLITEENGINE_H_
