#pragma once

#include <iostream>
#include <coroutine>
#include <fstream>
#include <__generator.hpp>  //reference implementation of generator
#include <torch/torch.h>
#include <sentencepiece_processor.h>
#include <rtg.hpp>
#include "utils.hpp"


namespace nn = torch::nn;
namespace optim = torch::optim;
namespace fs = std::filesystem;
namespace sp = sentencepiece;

using namespace std;
using namespace torch::indexing;
using namespace rtg;

namespace rtg::data {
    
    struct Example {

        int64_t id;
        vector<string> fields;
        vector<vector<int32_t>> field_ids;

        Example(int64_t id, vector<string> fields, vector<vector<int32_t>> field_ids)
        : id(id), fields(fields), field_ids(field_ids) {}

        ~Example() {}
        
        //copy ctr
        Example(const Example& other)
        : id(other.id), fields(other.fields), field_ids(other.field_ids) {}

        //copy assignment
        Example& operator=(const Example& other) {
            if (this != &other) {
                id = other.id;
                fields = other.fields;
                field_ids = other.field_ids;
            }
            return *this;
        }

        // move ctr
        Example(Example&& other) noexcept:
         id(other.id), fields(other.fields), field_ids(other.field_ids) {
        }

        friend std::ostream& operator<<(std::ostream& os, const Example& example);
       
    };

    // operator<<
    std::ostream& operator<<(std::ostream& os, const Example& ex) {
        os << "Example(" << ex.id << "; fields(" << 
        ex.fields.size() << "): " << ex.fields 
        << "; ids: (" << ex.field_ids.size() << "))";
        return os;
    }

    struct Batch {

        vector<torch::Tensor> fields;
        vector<Example> examples;

        Batch(vector<torch::Tensor> fields, vector<Example> examples)
        : fields(fields), examples(examples) {}
    
        static Batch from_buffer(vector<Example> buffer){
            /**
             * Convert a buffer of Examples to a Batch
            */
            int32_t batch_size = buffer.size();
            assert (batch_size > 0);
            int32_t num_fields = buffer[0].field_ids.size();

            vector<int32_t> max_lens(num_fields, 0);
            for (const auto& ex : buffer) {
                for (size_t i = 0; i < num_fields; ++i) {
                    max_lens[i] = std::max(max_lens[i], (int32_t)ex.field_ids[i].size());
                }
            }

            vector<torch::Tensor> fields(num_fields);
            int64_t pad_id = 0;  // pad token id
            for (size_t i = 0; i < num_fields; ++i) {
                fields[i] = torch::full({batch_size, max_lens[i]}, pad_id, torch::kLong);
                for (int32_t j = 0; j < buffer.size(); ++j) {
                    auto ids = torch::tensor( buffer[j].field_ids[i], torch::kLong);
                    fields[i].index_put_({j, Slice(0,  buffer[j].field_ids[i].size())}, ids);
                }
            }
            return Batch(fields, buffer);
        }

        auto to(torch::Device& device) -> Batch& {
            for(auto idx = 0; idx < fields.size(); ++idx) {
                fields[idx] = fields[idx].to(device);
            }
            return *this;
        }
        
        ~Batch() {}
    };
}