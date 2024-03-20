/**
 * @file yaml.hpp
 * @author Joshua Spisak (joshs333@live.com)
 * @brief Definitions to allow container to read / write to yaml with yaml-cpp.
 * @date 2023-05-23
 */
#ifndef JUMP_YAML_HPP_
#define JUMP_YAML_HPP_

// JUMP
#include <jump/memory_buffer.hpp>
#include <jump/array.hpp>
#include <jump/multi_array.hpp>
#include <jump/parallel.hpp>

// MISC
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>

namespace YAML {

/**
 * @brief encodes / decodes an array to yaml
 * @tparam array_t the underlying array type
 */
template<typename array_t>
struct convert<jump::array<array_t>> {
    /**
     * @brief encode an array to a yaml node
     * @param rhs the array to encode
     * @return the resulting yaml node
     */
    static Node encode(const jump::array<array_t>& rhs) {
        Node node;
        for(auto i = 0; i < rhs.size(); ++i) {
            node[i] = rhs.at(i);
        }
        return node;
    }

    /**
     * @brief decode a yaml node to an array
     * @param node the node to decode
     * @param rhs the array to decode into
     * @return whether or not it successfully converted it
     */
    static bool decode(const Node& node, jump::array<array_t>& rhs) {
        if(!node.IsSequence()) {
            return false;
        }

        rhs = jump::array<array_t>(node.size());

        for(std::size_t i = 0; i < node.size(); ++i)
            rhs[i] = node[i].as<array_t>();
        return true;
    }

}; /* struct convert<jump::array<array_t>> */

/**
 * @brief encodes / decodes indices to yaml
 * @tparam _)max_dims the max dimensionality of teh indices
 */
template<std::size_t _max_dims>
struct convert<jump::multi_indices<_max_dims>> {
    /**
     * @brief encode indices to a yaml node
     * @param rhs the indices to encode
     * @return the resulting yaml node
     */
    static Node encode(const jump::multi_indices<_max_dims>& rhs) {
        Node node;
        for(auto i = 0; i < rhs.dims(); ++i) {
            node[i] = rhs[i];
        }
        return node;
    }

    /**
     * @brief decode a yaml node to indices
     * @param node the node to decode
     * @param rhs the indices to decode into
     * @return whether or not it successfully converted it
     */
    static bool decode(const Node& node, jump::multi_indices<_max_dims>& rhs) {
        if(!node.IsSequence()) {
            return false;
        }

        rhs.dims() = node.size();

        for(std::size_t i = 0; i < node.size(); ++i)
            rhs[i] = node[i].as<std::size_t>();
        return true;
    }

}; /* struct convert<jump::multi_indices<_max_dims>> */

/**
 * @brief encodes / decodes a multi_array to yaml
 * @tparam array_t the underlying array type
 * @tparam _max_dims the maximum dimensionality of the array
 */
template<typename array_t, std::size_t _max_dims>
struct convert<jump::multi_array<array_t, _max_dims>> {
    //! Multi-ARray-TYpe, I'm so punny.
    using marty = jump::multi_array<array_t, _max_dims>;

    //! Recurse through the dimensions of the multi_array to encode it
    static void encodeHelper(
        Node& node,
        const marty& rhs,
        typename marty::indices indices,
        std::size_t depth = 0
    ) {
        for(std::size_t i = 0; i < rhs.shape(depth); ++i) {
            indices[depth] = i;
            if(depth >= rhs.dims() - 1) {
                node[i] = rhs.at(indices);
            } else {
                YAML::Node n;
                encodeHelper(n, rhs, indices, depth + 1);
                node[i] = std::move(n);
            }
        }
    }

    /**
     * @brief encode an multi_array to a yaml node
     * @param rhs the multi_array to encode
     * @return the resulting yaml node
     */
    static Node encode(const marty& rhs) {
        Node node;
        node["shape"] = rhs.shape();
        Node data;
        encodeHelper(data, rhs, rhs.zero(), 0);
        node["data"] = std::move(data);
        return node;
    }

    /**
     * @brief recurse through each sub-sequence to determine the dimensions
     *  of the multi-array this node encodes
     * @param node the node to to size up
     * @param size the resulting size (in indices)
     * @param depth the depth we are at (is a recursive call)
     */
    static void sizeHelper(
        const Node& node,
        typename marty::indices& size,
        std::size_t depth = 0
    ) {
        // if we reach the end, we mark depth and return
        if(!node.IsSequence()) {
            size.dims() = depth;
            return;
        }
        // for each of the members we recurse and see which has the highest size
        auto next_size = size;
        for(std::size_t i = 0; i < node.size(); ++i) {
            sizeHelper(node[i], next_size, depth + 1);
            if(next_size.dims() == depth + 1) {
                size = next_size;
                break;
            } else {
                if(next_size[depth + 1] > size[depth + 1])
                    size = next_size;
                    continue;
            }
        }
        size[depth] = node.size();
        return;
    }

    /**
     * @brief recurse through a yaml node to extract all the values
     * @param node the node to decode from
     * @param arr the array to decode into
     * @param index index tracking
     * @param depth the depth (is a recursive call)
     */
    static void decodeHelper(
        const Node& node,
        marty& arr,
        typename marty::indices index,
        std::size_t depth = 0
    ) {
        for(std::size_t i = 0; i < node.size(); ++i) {
            index[depth] = i;
            if(node[i].IsSequence() && depth + 1 < arr.dims()) {
                decodeHelper(node[i], arr, index, depth + 1);
            } else {
                arr.at(index) = node[i].as<array_t>();
            }
        }
    }

    /**
     * @brief decode a yaml node to a multi_array
     * @param node the node to decode
     * @param rhs the multi_array to decode into
     * @return whether or not it successfully converted it
     */
    static bool decode(const Node& node, marty& rhs) {
        auto shape = node["shape"].as<jump::multi_indices<_max_dims>>();
        marty result(shape);
        decodeHelper(node["data"], result, result.zero());
        rhs = std::move(result);
        return true;
    }

}; /* struct convert<jump::multi_array<array_t, _max_dims>> */

/**
 * @brief define how to convert between eigen matrices and YAML
 * @tparam matrix_t the underlying matrix type
 * @tparam size_a matrix dimension
 * @tparam size_b matrix dimension
 * @tparam f1 template field for compatibility
 * @tparam f2 template field for compatibility
 * @tparam f3 template field for compatibility
 */
template<typename matrix_t, int size_a, int size_b, int f1, int f2, int f3>
struct convert<Eigen::Matrix<matrix_t, size_a, size_b, f1, f2, f3>> {
    //! Matrix type definition for convenience
    using mat = Eigen::Matrix<matrix_t, size_a, size_b, f1, f2, f3>;

    /**
     * @brief encode an array to a yaml node
     * @param rhs the array to encode
     * @return the resulting yaml node
     */
    static Node encode(const mat& rhs) {
        Node node;
        for(std::size_t i = 0; i < rhs.rows(); ++i) {
            Node row;
            for(std::size_t j = 0; j < rhs.cols(); ++j) {
                row[j] = rhs(i, j);
            }
            node[i] = row;
        }
        return node;
    }

    /**
     * @brief decode a yaml node to an array
     * @param node the node to decode
     * @param rhs the array to decode into
     * @return whether or not it successfully converted it
     */
    static bool decode(const Node& node, mat& rhs) {
        if(!node.IsSequence()) {
            return false;
        }
        for(std::size_t i = 0; i < rhs.rows() && i < node.size(); ++i) {
            if(!node[i].IsSequence()) {
                rhs(i, 0) = node[i].as<matrix_t>();
            } else {
                for(std::size_t j = 0; j < rhs.cols() && j < node[i].size(); ++j) {
                    rhs(i, j) = node[i][j].as<matrix_t>();
                }
            }
        }

        return true;
    }

}; /* struct convert<Eigen::Matrix<matrix_t, size_a, size_b, f1, f2, f3>> */

/**
 * @brief convert memory_t to yaml / back
 */
template<>
struct convert<jump::memory_t> {
    //! Encode memory_t to yaml
    static Node encode(const jump::memory_t& rhs) {
        YAML::Node result;
        result = std::string(jump::memory_t_str[static_cast<std::size_t>(rhs)]);
        return result;
    }

    //! Decode memory_t from yaml (return true / false on success)
    static bool decode(const Node& node, jump::memory_t& rhs) {
        rhs = jump::memory_t::UNKNOWN;
        for(std::size_t i = 0; i < static_cast<std::size_t>(jump::memory_t::UNKNOWN); ++i) {
            if(jump::memory_t_str[i] == node.as<std::string>()) {
                rhs = static_cast<jump::memory_t>(i);
            }
        }
        return true;
    }

}; /* struct convert<jump::memory_t> */

/**
 * @brief convert jump::par::target_t to yaml / back
 */
template<>
struct convert<jump::par::target_t> {
    //! Encode memory_t to yaml
    static Node encode(const jump::par::target_t& rhs) {
        YAML::Node result;
        result = std::string(jump::par_target_t_strs[static_cast<std::size_t>(rhs)]);
        return result;
    }

    //! Decode memory_t from yaml (return true / false on success)
    static bool decode(const Node& node, jump::par::target_t& rhs) {
        rhs = jump::par::target_t::unknown;
        for(std::size_t i = 0; i < static_cast<std::size_t>(jump::par::target_t::unknown); ++i) {
            if(jump::par_target_t_strs[i] == node.as<std::string>()) {
                rhs = static_cast<jump::par::target_t>(i);
            }
        }
        return true;
    }

}; /* struct convert<jump::memory_t> */

/**
 * @brief conver jump::par to yaml / back
 */
template<>
struct convert<jump::par> {
    //! Encode memory_t to yaml
    static Node encode(const jump::par& rhs) {
        YAML::Node result;
        result["target"] = rhs.target;
        result["thread_count"] = rhs.thread_count;
        result["threads_per_block"] = rhs.threads_per_block;
        result["disable_device_transfers"] = rhs.disable_device_transfers;
        result["debug"] = rhs.debug;
        return result;
    }

    //! Decode memory_t from yaml (return true / false on success)
    static bool decode(const Node& node, jump::par& rhs) {
        if(node["target"])
            rhs.target = node["target"].as<jump::par::target_t>();
        if(node["thread_count"])
            rhs.thread_count = node["thread_count"].as<std::size_t>();
        if(node["threads_per_block"])
            rhs.threads_per_block = node["threads_per_block"].as<std::size_t>();
        if(node["disable_device_transfers"])
            rhs.disable_device_transfers = node["disable_device_transfers"].as<bool>();
        if(node["debug"])
            rhs.debug = node["debug"].as<bool>();
        return true;
    }
};

} /* namespace YAML */

#endif /* JUMP_YAML_HPP_ */
