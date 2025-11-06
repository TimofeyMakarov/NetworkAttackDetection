#include "ONNXModel.h"
#include <iostream>

ONNXModel::ONNXModel(const ORTCHAR_T* model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "NetworkIDS"),
    session(env, model_path, Ort::SessionOptions{}) {

    Ort::AllocatorWithDefaultOptions allocator;

    // Кол-во возможных входов
    size_t num_input_nodes = session.GetInputCount();

    for (size_t i = 0; i < num_input_nodes; i++) {
        // Получаем имя входа
        Ort::AllocatedStringPtr input_name_ptr = session.GetInputNameAllocated(i, allocator);
        input_name_ptrs.push_back(std::move(input_name_ptr));
        input_names.push_back(input_name_ptrs[i].get());

        // Информация о типе ожидаемых данных и размере
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        input_shapes.push_back(shape);
    }

    // Кол-во возможных выходов
    size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++) {
        // Получаем имена выходов модели
        Ort::AllocatedStringPtr output_name_ptr = session.GetOutputNameAllocated(i, allocator);
        output_name_ptrs.push_back(std::move(output_name_ptr));
        output_names.push_back(output_name_ptrs[i].get());
    }

    std::cout << "Model was loaded" << std::endl;
}

// Метод для предсказания для одного объекта
std::vector<int64_t> ONNXModel::predict(std::vector<float>& input_data) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Создаем входной тензор
    std::vector<int64_t> input_shape = input_shapes[0];

    // Заменяем динамическую размерность (-1) на 1
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
        input_data.data(),   // Сами данные
        input_data.size(),   // Размер данных
        input_shape.data(),  // Информация о форме данных
        input_shape.size()   // Размерность 
    );

    // Âûïîëíÿåì inference
    auto output_tensors = session.Run(
        Ort::RunOptions{ nullptr }, // Опции выполнения (nullptr - настройки по умолчанию)
        input_names.data(),         // Имена входов
        &input_tensor,              // Входные тензоры
        input_names.size(),         // Количество входов
        output_names.data(),        // Имена выходов
        output_names.size()         // Количество выходов
    );

    // Обрабатываем результаты
    const int64_t* output_data = output_tensors[0].GetTensorData<int64_t>();
    auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();

    return std::vector<int64_t>(output_data, output_data + output_size);
}

// Метод для предсказания для нескольких объектов
std::vector<int64_t> ONNXModel::predict(std::vector<std::vector<float>>& input_data) {
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    // Создаем входной тензор
    std::vector<int64_t> input_shape = input_shapes[0];

    // Заменяем динамическую размерность на реальный batch_size
    size_t batch_size = input_data.size();
    for (size_t i = 0; i < input_shape.size(); i++) {
        if (input_shape[i] == -1) {
            input_shape[i] = batch_size;
            break;
        }
    }

    // Проверяем, что все samples имеют одинаковый размер
    size_t features_count = input_data[0].size();
    if (features_count != 78) {
        throw std::runtime_error("Expected 78 features, got " + std::to_string(features_count));
    }
    for (const auto& sample : input_data) {
        if (sample.size() != features_count) {
            throw std::runtime_error("All samples must have same number of features");
        }
    }

    // Преобразуем input_data в std::vector<float>
    std::vector<float> input_data_processed;
    input_data_processed.reserve(batch_size * features_count);
    for (const auto& features : input_data) {
        input_data_processed.insert(input_data_processed.end(), features.begin(), features.end());
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data_processed.data(),   // Сами данные
        input_data_processed.size(),   // Размер данных
        input_shape.data(),            // Информация о форме данных
        input_shape.size()             // Размерность 
    );

    // Выполняем inference
    auto output_tensors = session.Run(
        Ort::RunOptions{ nullptr }, // Опции выполнения (nullptr - настройки по умолчанию)
        input_names.data(),         // Имена входов
        &input_tensor,              // Входные тензоры
        input_names.size(),         // Количество входов
        output_names.data(),        // Имена выходов
        output_names.size()         // Количество выходов
    );

    // Обрабатываем результаты
    const int64_t* output_data = output_tensors[0].GetTensorData<int64_t>();
    auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t output_size = tensor_info.GetElementCount();

    return std::vector<int64_t>(output_data, output_data + output_size);
}
