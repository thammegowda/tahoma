#include <iostream>
#include <tahoma.hpp>

// rename main as trainer_main 
#define main trainer_main
#include "train/trainer.hpp"
#undef main


int main(int argc, char* argv[]) {
    using namespace tahoma;
    int _code = global_setup();
    if (_code != 0){
        return _code;
    }
    return trainer_main(argc, argv);
}