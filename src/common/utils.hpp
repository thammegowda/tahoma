#include<fstream>
#include<filesystem>
#include<__generator.hpp>


namespace rtg::utils {

    std::generator<std::string> read_lines(std::string path){
        /**
         * Read lines from a file and yield them one by one.
        */
        std::ifstream file(path);
        std::string line;
        while(std::getline(file, line)){
            co_yield line;
        }
        file.close();
    }
}