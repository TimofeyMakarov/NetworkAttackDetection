#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "ONNXModel.h"

std::vector<std::vector<float>> read_test_data(const std::string& filename) {
    std::vector<std::vector<float>> data;
    std::ifstream file(filename);
    std::string line;
    size_t line_count = 0;

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    while (std::getline(file, line)) {
        line_count++;
        std::vector<float> row;
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, ',')) {
            row.push_back(std::stof(value));
        }

        // Проверяем, что в строке 79 значений (78 features + 1 target)
        if (row.size() == 79) {
            data.push_back(row);
        }
        else {
            std::cout << "Incorrect line: " << line_count  << std::endl;
        }
    }

    file.close();
    return data;
}

bool has_cyrillic(const std::string& str) {
    for (char c : str) {
        if ((c >= 'А' && c <= 'я') || c == 'ё' || c == 'Ё') {
            return true;
        }
    }
    return false;
}

int main(int argc, char* argv[]) {
    try {
        // Используем PROJECT_DIR определенный в CMake
        std::string project_dir = PROJECT_DIR;
        if (has_cyrillic(project_dir)) {
            std::cerr << "ERROR: Project path contains Cyrillic characters!" << std::endl;
            std::cerr << "Current path: " << project_dir << std::endl;
            std::cerr << "Please move the project to a path without Cyrillic letters and rebuild it." << std::endl;
            return 1;
        }

        
        std::string model_name = "DecisionTree_CICIDS2017_NetworkAttackDetector_v1.onnx";
        std::string data_name = "sample_data.csv";

        if (argc > 1) model_name = argv[1];
        if (argc > 2) data_name = argv[2];

        // Формируем пути
        std::string model_path = project_dir + "/models/" + model_name;
        std::string data_path = project_dir + "/data/" + data_name;

        // Конвертируем в wstring для модели
        std::wstring wmodel_path(model_path.begin(), model_path.end());

        // Загружаем модель
        ONNXModel model(wmodel_path);

        // Тестовые данные
        std::vector<std::vector<float>> X_test;
        std::vector<int64_t> y_test;
        X_test = read_test_data(data_path);
        std::cout << "Loaded " << X_test.size() << " test samples" << std::endl;

        // Для accuracy
        size_t correct_predictions = 0;
        size_t total_predictions = 0;

        for (std::vector<float>& sample : X_test) {
            // Берем последний элемент (быстрее чем первый)
            y_test.push_back(static_cast<int64_t>(sample.back()));
            sample.pop_back();
        }

        std::vector<int64_t> y_pred = model.predict(X_test);
        for (size_t i = 0; i < y_test.size(); i++) {
            if (y_test[i] == y_pred[i]) {
                correct_predictions++;
            }
            total_predictions++;
        }

        // Выводим результат
        float accuracy = static_cast<float>(correct_predictions) / total_predictions;
        std::cout << "Accuracy: " << accuracy << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}