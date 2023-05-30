// JUMP
#include <jump/array.hpp>
#include <jump/multi_array.hpp>
#include <jump/yaml.hpp>

// MISC
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>

// STD
#include <iostream>
#include <fstream>
#include <cstdio>

int main(int argc, char** argv) {
    jump::array<int> arr_data(10, 10);
    jump::multi_array<int> multi_arr_data({2, 2, 3}, 4);
    jump::multi_array<Eigen::Matrix<double, 2, 2>> multi_arr_eigen({3, 3}, Eigen::Matrix<double, 2, 2>::Zero());
    multi_arr_eigen.at(1, 1) << 1, 2, 3, 4;
    multi_arr_data.at(1, 1, 1) = 5;

    Eigen::Matrix<double, 3, 1> mat;
    mat << 1, 2, 3;

    YAML::Node yaml_data;
    // yaml_data << arr_data;
    yaml_data["array"] = arr_data;
    yaml_data["marray"] = multi_arr_data;
    yaml_data["mat"] = mat;
    yaml_data["marray_eigen"] = multi_arr_eigen;

    auto a = yaml_data["marray"].as<jump::multi_array<int>>();
    std::printf("A(1,1,1) %d\n", a.at(1, 1, 1));
    std::printf("A(1,1,2) %d\n", a.at(1, 1, 2));
    auto b = yaml_data["marray"].as<jump::multi_array<std::size_t>>();
    std::printf("B(1,1,1) %d\n", a.at(1, 1, 1));
    std::printf("B(1,1,2) %d\n", a.at(1, 1, 2));
    auto c = yaml_data["marray_eigen"].as<jump::multi_array<Eigen::Matrix<double, 2, 2>>>();
    std::cout << c.at(1, 1) << std::endl;

    std::ofstream fout("dummy.yaml");
    fout << yaml_data;
    return 0;
}
