#include <tahoma.h>
#include <tahoma/model.h>
#include <tahoma/train/trainer.h>

namespace tahoma::tests {

    auto test_trainer_nmt(std::vector<std::string> args) -> int {
        if (args.size() != 2) {
            std::cerr << "Error: test_trainer_nmt requires 2 arguments, but got " << args.size() << "\n";
            std::cerr << "Usage: test_trainer_nmt <config_file> <work_dir>\n";
            return 1;
        }
        auto config_file = args[0];
        auto work_dir = args[1];
        auto config = tahoma::config::Config(config_file);
        auto trainer = tahoma::train::Trainer(work_dir, config);
        trainer.train();
        return 0;
    }

}