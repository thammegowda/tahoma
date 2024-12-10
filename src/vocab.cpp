#include <tahoma/vocab.h>
#include <sentencepiece_processor.h>


namespace tahoma {

    SPVocab::SPVocab (const std::string& vocab_path) {
        spdlog::info("Loading vocab from {}", vocab_path);
        spp = std::make_shared<sentencepiece::SentencePieceProcessor>();
        if (!fs::exists(vocab_path)) {
            throw std::runtime_error("Vocab file " + vocab_path + " not found");
        }
        if (!spp->Load(vocab_path).ok()) {
            throw std::runtime_error("Unable to load vocab from " + vocab_path);
        }
    }

    auto SPVocab::encode(const std::string& text) -> std::vector<size_t> {
        auto seq = spp->EncodeAsIds(text);
        return std::vector<size_t>(seq.begin(), seq.end());
    }

    auto SPVocab::decode(const std::vector<size_t>& ids) -> std::string {
        vector<i32> seq = std::vector<i32>(ids.begin(), ids.end());
        return spp->DecodeIds(seq);
    }

}