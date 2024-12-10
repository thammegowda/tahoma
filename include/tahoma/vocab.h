#pragma once

#include <tahoma.h>
#include <sentencepiece_processor.h>

namespace tahoma {

    class IVocab {
    public:
        virtual auto encode(const std::string& text) -> std::vector<size_t> = 0;
        virtual auto decode(const std::vector<size_t>& ids) -> std::string = 0;
        virtual auto pad_idx() -> size_t = 0;

        // default constructor
        IVocab() = default;
        // virtual destructor
        ~IVocab() = default;
        // Copy constructor
        IVocab(const IVocab& other) = default;
        // Move constructor
        IVocab(IVocab&& other) noexcept = default;
        // Copy assignment operator
        IVocab& operator=(const IVocab& other) = default;
        // Move assignment operator
        IVocab& operator=(IVocab&& other) noexcept = default;


        auto encode_batch(const std::vector<std::string>& texts, torch::Device device=torch::kCPU) -> Tensor {
            std::vector<std::vector<size_t>> result;
            size_t max_len = 0;
            for (const auto& text : texts) {
                result.push_back(encode(text));
                max_len = std::max(max_len, result.back().size());
            }
            auto tensor = torch::full({(i64) texts.size(), (i64) max_len}, pad_idx(),
                 torch::dtype(torch::kLong).device(device));
            for (size_t i = 0; i < texts.size(); ++i) {
                // TODO: bulk copy instead of loop
                for (size_t j = 0; j < result[i].size(); ++j) {
                    tensor[i][j] = result[i][j];
                }
            }
            return tensor;
        }
        virtual auto size() -> int = 0;
    };

    class SPVocab : public IVocab {
    protected:
        std::shared_ptr<sentencepiece::SentencePieceProcessor> spp;
    public:
        SPVocab(const std::string& vocab_path);
        
        // Copy constructor
        //SPVocab(const SPVocab& other) : spp(other.spp) {}

        // Move constructor
        //SPVocab(SPVocab&& other) noexcept : spp(std::move(other.spp)) {}

        // Copy assignment operator
        SPVocab& operator=(const SPVocab& other) {
            if (this != &other) {
                spp = other.spp;
            }
            return *this;
        }

        // Move assignment operator
        SPVocab& operator=(SPVocab&& other) noexcept {
            if (this != &other) {
            spp = std::move(other.spp);
            }
            return *this;
        }

        auto encode(const std::string& text) -> std::vector<size_t> override;
        auto decode(const std::vector<size_t>& ids) -> std::string override;

        auto size() -> int override {
            return spp->GetPieceSize();
        }
    };

}