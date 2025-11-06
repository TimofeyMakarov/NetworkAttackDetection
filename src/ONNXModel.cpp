#include "ONNXModel.h"
#include <iostream>

ONNXModel::ONNXModel(const ORTCHAR_T* model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "NetworkIDS"),
    session(env, model_path, Ort::SessionOptions{}) {

    Ort::AllocatorWithDefaultOptions allocator;

    // Êîë-âî âîçìîæíûõ âõîäîâ
    size_t num_input_nodes = session.GetInputCount();

    for (size_t i = 0; i < num_input_nodes; i++) {
        // Ïîëó÷àåì èìÿ âõîäà
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(i, allocator);
        input_name_ptrs.push_back(std::move(input_name_ptr));
        input_names.push_back(input_name_ptrs[i].get());

        // Èíôîðìàöèÿ î òèïå îæèäàåìûõ äàííûõ è ðàçìåðå
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        input_shapes.push_back(shape);
    }

    // Êîë-âî âîçìîæíûõ âûõîäîâ
    size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        // Ïîëó÷àåì èìåíà âûõîäîâ ìîäåëè
        Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(i, allocator);
        output_name_ptrs.push_back(std::move(output_name_ptr));
        output_names.push_back(output_name_ptrs[i].get());
    }

    std::cout << "Model was loaded" << std::endl;
}

// Ìåòîä äëÿ ïðåäñêàçàíèÿ äëÿ îäíîãî îáúåêòà
std::vector<int64_t> ONNXModel::predict(std::vector<float>& input_data) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Ñîçäàåì âõîäíîé òåíçîð
    std::vector<int64_t> input_shape = input_shapes[0];

    // Çàìåíÿåì äèíàìè÷åñêóþ ðàçìåðíîñòü (-1) íà 1
    size_t total_elements = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
        if (input_shape[i] == -1) {
            input_shape[i] = 1;
        }
        total_elements *= input_shape[i];
    }

    if (input_data.size() != total_elements) {
        throw std::runtime_error("Input data size doesn't match expected shape");
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),   // Ñàìè äàííûå
        input_data.size(),   // Ðàçìåð äàííûõ
        input_shape.data(),  // Èíôîðìàöèÿ î ôîðìå äàííûõ
        input_shape.size()   // Ðàçìåðíîñòü 
    );

    // Âûïîëíÿåì inference
    auto output_tensors = session.Run(
        Ort::RunOptions{ nullptr }, // Îïöèè âûïîëíåíèÿ (nullptr - íàñòðîéêè ïî óìîë÷àíèþ)
        input_names.data(),         // Èìåíà âõîäîâ
        &input_tensor,              // Âõîäíûå òåíçîðû
        input_names.size(),         // Êîëè÷åñòâî âõîäîâ
        output_names.data(),        // Èìåíà âûõîäîâ
        output_names.size()         // Êîëè÷åñòâî âûõîäîâ
    );

    // Îáðàáàòûâàåì ðåçóëüòàòû
    const int64_t* output_data = output_tensors[0].GetTensorData<int64_t>();
    auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();

    return std::vector<int64_t>(output_data, output_data + output_size);
}

// Ìåòîä äëÿ ïðåäñêàçàíèÿ äëÿ íåñêîëüêèõ îáúåêòîâ
std::vector<int64_t> ONNXModel::predict(std::vector<std::vector<float>>& input_data) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Ñîçäàåì âõîäíîé òåíçîð
    std::vector<int64_t> input_shape = input_shapes[0];

    // Çàìåíÿåì äèíàìè÷åñêóþ ðàçìåðíîñòü íà ðåàëüíûé batch_size
    size_t batch_size = input_data.size();
    for (size_t i = 0; i < input_shape.size(); i++) {
        if (input_shape[i] == -1) {
            input_shape[i] = batch_size;
            break;
        }
    }

    // Ïðîâåðÿåì, ÷òî âñå samples èìåþò îäèíàêîâûé ðàçìåð
    size_t features_count = input_data[0].size();
    if (features_count != 78) {
        throw std::runtime_error("Expected 78 features, got " + std::to_string(features_count));
    }
    for (const auto& sample : input_data) {
        if (sample.size() != features_count) {
            throw std::runtime_error("All samples must have same number of features");
        }
    }

    // Ïðåîáðàçóåì input_data â std::vector<float>
    std::vector<float> input_data_processed;
    input_data_processed.reserve(batch_size * features_count);
    for (const auto& features : input_data) {
        input_data_processed.insert(input_data_processed.end(), features.begin(), features.end());
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data_processed.data(),   // Ñàìè äàííûå
        input_data_processed.size(),             // Ðàçìåð äàííûõ
        input_shape.data(),            // Èíôîðìàöèÿ î ôîðìå äàííûõ
        input_shape.size()             // Ðàçìåðíîñòü 
    );

    // Âûïîëíÿåì inference
    auto output_tensors = session.Run(
        Ort::RunOptions{ nullptr }, // Îïöèè âûïîëíåíèÿ (nullptr - íàñòðîéêè ïî óìîë÷àíèþ)
        input_names.data(),         // Èìåíà âõîäîâ
        &input_tensor,              // Âõîäíûå òåíçîðû
        input_names.size(),         // Êîëè÷åñòâî âõîäîâ
        output_names.data(),        // Èìåíà âûõîäîâ
        output_names.size()         // Êîëè÷åñòâî âûõîäîâ
    );

    // Îáðàáàòûâàåì ðåçóëüòàòû
    const int64_t* output_data = output_tensors[0].GetTensorData<int64_t>();
    auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();

    return std::vector<int64_t>(output_data, output_data + output_size);
}
