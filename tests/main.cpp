// unit tests for tahoma
// created by TG; circa 2024 fall-winter
//
#include <iostream>

#include "test_data_loader.cpp"
#include "test_serialize.cpp"
#include "test_trainer.cpp"
#include "test_parallel_init.cpp"

using namespace tahoma::tests;

using TestFunc = std::function<int(std::vector<std::string>)>;

struct TestCase {
    TestFunc func;
    std::string name;
    int nargs;
    std::string description;
    std::string help;

    TestCase(TestFunc func, std::string name, int nargs, const std::string description, const std::string help = "")
        : func(func), name(name), nargs(nargs), description(description), help(help) {
    }

    TestCase() = default;
    TestCase(const TestCase&) = default;
    TestCase(TestCase&&) = default;
    TestCase& operator=(const TestCase&) = default;
    TestCase& operator=(TestCase&&) = default;
    ~TestCase() = default;

    int operator()(std::vector<std::string> args) {
        if (args.size() != nargs) {
            std::cerr << "Error: " << name << " requires " << nargs << " arguments, but got " << args.size() << "\n";
            std::cerr << "Usage: " << name << " " << help << "\n";
            return 1;
        }
        return func(args);
    }

    friend std::ostream& operator<<(std::ostream& os, const TestCase& meta) {
        os << meta.name << " :\n    " << meta.help << "\n    Desc: " << meta.description;
        return os;
    }
};


void usage(const std::string& program_name, const std::vector<TestCase>& index) {
    std::cerr << "Usage: " << program_name << " <test_name> <args...>\n\nAvailable tests:\n";
    for (const auto& meta : index) {
        std::cerr << meta << "\n";
    }
}


int main(int argc, char const* argv[]) {
    std::vector<std::string> args(argv, argv + argc);
    tahoma::global_setup();
    std::string program_name = argv[0];
    std::vector<TestCase> tests = {
        {test_config_parse, "config_parse", 1, "Parse a config file", "Args: <config_file>"},
        {test_read_lines, "read_lines", 1, "Read lines from a file", "Args: <config_file>"},
        {test_read_examples, "read_examples", 1, "Read examples from a file", "Args: <config_file>"},
        {test_make_batches, "make_batches", 1, "Read examples from a file", "Args: <config_file>"},
        {test_data_loader_sync, "data_loader_sync", 1, "Load data from a config file", "Args: <config_file>"},
        {test_data_loader_async, "data_loader_async", 1, "Load data from a config file", "Args: <config_file>"},
        {test_samples, "samples", 1, "Load samples from a config file", "Args: <config_file>"},
        {test_npz_load, "npz_load", 1, "Load model from checkpt file", "Args: <checkpt.npz>"},
        {test_trainer_nmt, "trainer_nmt", 2, "Train a model", "Args: <config_file> <work_dir>"},
        {test_parallel_init, "parallel_init", 1, "Test model initialization time", "Args: <config_file>"},
    };
    std::map<std::string, TestCase> index = {};
    for (const auto& test : tests) {
        index[test.name] = test;
    }
    if (argc < 2) {
        std::cerr << "Error: test name required\n";
        usage(program_name, tests);
        return 1;
    }

    std::string test_name = argv[1];
    if (test_name == "-h" || test_name == "--help" || test_name == "help") {
        usage(program_name, tests);
        return 0;
    }
    if (index.count(test_name) == 0) {
        std::cerr << "Error:: unknown test '" << test_name << "'\n";
        usage(program_name, tests);
        return 1;
    } else {
        args = std::vector<std::string>(args.begin() + 2, args.end());
        return index[test_name](args);
    }
}
