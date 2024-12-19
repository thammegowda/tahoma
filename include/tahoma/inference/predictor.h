#pragma once

#include <tahoma.h>
#include <tahoma/data.h>
#include <tahoma/model.h>

using namespace tahoma;

namespace tahoma::inference {

    void predict_scores(Ptr<model::LanguageModel> model, data::DataLoader& loader, string file_name, Pack kwargs);

    void predict(string model_path, string input_file, Pack kwargs);

} // namespace tahoma::inference