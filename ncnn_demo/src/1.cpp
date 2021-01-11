// //#include <torch/script.h> // One-stop header.

// #include <iostream>
// #include <memory>

// #include <sys/time.h>

// int main(int argc, const char* argv[]) {

// std::string model_file = "/home/ly/newdisk/workspace1/projects/rgbdml/fusionnet/320x240/mobile_best_model_320x240.pt";

// //  if (argc != 2) {
// //    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
// //    return -1;
// //  }

//  if (argc == 2) {
//      model_file = std::string(argv[1]);
//  }

//   torch::jit::script::Module module;
//   try {
//     // Deserialize the ScriptModule from a file using torch::jit::load().
//     module = torch::jit::load(model_file);
//   }
//   catch (const c10::Error& e) {
//     std::cerr << "error loading the model\n";
//     return -1;
//   }
//   // Create a vector of inputs.
//   std::vector<torch::jit::IValue> inputs;
// //  inputs.push_back(torch::ones({1, 3, 640, 480}));
// //  inputs.push_back(torch::ones({1, 1, 640, 480}));

//   inputs.push_back(torch::ones({1, 3, 320, 240}));
//   inputs.push_back(torch::ones({1, 1, 320, 240}));