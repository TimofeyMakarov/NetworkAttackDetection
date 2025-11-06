#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

class ONNXModel {
private:
    Ort::Env env;
    Ort::Session session;
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<std::vector<int64_t>> input_shapes;

public:
    ONNXModel(const ORTCHAR_T* model_path);
    std::vector<int64_t> predict(std::vector<std::vector<float>>& input_data); // äëÿ ïðåäñêàçàíèÿ ìíîæåñòâà samples
    std::vector<int64_t> predict(std::vector<float>& input_data); // äëÿ îäíîãî sample
};
