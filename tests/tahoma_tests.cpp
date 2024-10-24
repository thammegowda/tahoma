
#include <iostream>

#include "test_data_loader.hpp"
using namespace tahoma::tests;

using TestFunc = std::function<int(std::vector<std::string>)>;

struct TestMeta {
    TestFunc func;
    std::string name;
    int nargs;
    std::string description;
    std::string help;

    TestMeta(TestFunc func, std::string name, int nargs, const std::string description, const std::string help = "")
        : func(func), name(name), nargs(nargs), description(description), help(help) {}

    TestMeta() = default;
    TestMeta(const TestMeta&) = default;
    TestMeta(TestMeta&&) = default;
    TestMeta& operator=(const TestMeta&) = default;
    TestMeta& operator=(TestMeta&&) = default;
    ~TestMeta() = default;

    int operator()(std::vector<std::string> args) {
        if (args.size() != nargs) {
            std::cerr << "Error: " << name << " requires " << nargs << " arguments, but got " << args.size() << "\n";
            std::cerr << "Usage: " << name << " " << help << "\n";
            return 1;
        }
        return func(args);
    }

    friend std::ostream& operator<<(std::ostream& os, const TestMeta& meta) {
        os << meta.name << ": " << meta.description << "\n\t" << meta.help;
        return os;
    }
};


void usage(const std::string& program_name, const std::vector<TestMeta>& index) {
    std::cerr << "Usage: " << program_name << " <test_name> <args...>\nAvailable tests:\n";
    for (const auto&  meta : index) {
        std::cerr << meta << "\n";
    }
}


int main(int argc, char const* argv[]) {
    std::vector<std::string> args(argv, argv + argc);
    tahoma::global_setup();
    std::string program_name = argv[0];
    std::vector<TestMeta> tests = {
        TestMeta(test_config_parse, "config_parse", 1, "Parse a config file", "Args: <config_file>"),
        TestMeta(test_read_lines, "read_lines", 1, "Read lines from a file", "Args: <config_file>"),
        TestMeta(test_read_examples, "read_examples", 1, "Read examples from a file", "Args: <config_file>"),
        TestMeta(test_data_loader_sync, "data_loader_sync", 1, "Load data from a config file", "Args: <config_file>"),
        TestMeta(test_data_loader_async, "data_loader_async", 1, "Load data from a config file", "Args: <config_file>"),
        TestMeta(test_samples, "samples", 1, "Load samples from a config file", "Args: <config_file>"),
    };
    std::map<std::string, TestMeta> index = {};
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
